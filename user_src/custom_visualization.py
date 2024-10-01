from typing import List, Optional
import numpy as np
import torch
import cv2
import mmcv

from mmdet.visualization.local_visualizer import DetLocalVisualizer
from mmdet.visualization.palette import _get_adaptive_scales, get_palette, jitter_color
from mmdet.structures.mask.structures import BitmapMasks, PolygonMasks, bitmap_to_polygon
from mmdet.registry import VISUALIZERS

from mmengine.structures import InstanceData
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.dist import master_only


def bbox_iou(boxA, boxB):  
    xA = torch.max(boxA[0], boxB[0])  
    yA = torch.max(boxA[1], boxB[1])  
    xB = torch.min(boxA[2], boxB[2])  
    yB = torch.min(boxA[3], boxB[3])  
  
    interArea = torch.max(torch.tensor(0), xB - xA) * torch.max(torch.tensor(0), yB - yA)  
  
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])  
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  
  
    iou = interArea / (boxAArea + boxBArea - interArea)  
  
    return iou 

@VISUALIZERS.register_module()
class TwoStreamDetLocalVisualizer(DetLocalVisualizer):
    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances and instances.bboxes.sum() > 0:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)

            positions = bboxes[:, :2] + self.line_width
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                if 'label_names' in instances:
                    label_text = instances.label_names[i]
                else:
                    label_text = classes[
                        label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

            if len(labels) > 0 and \
                    ('bboxes' not in instances or
                     instances.bboxes.sum() == 0):
                # instances.bboxes.sum()==0 represent dummy bboxes.
                # A typical example of SOLO does not exist bbox branch.
                areas = []
                positions = []
                for mask in masks:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8)
                    if stats.shape[0] > 1:
                        largest_id = np.argmax(stats[1:, -1]) + 1
                        positions.append(centroids[largest_id])
                        areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)

                for i, (pos, label) in enumerate(zip(positions, labels)):
                    if 'label_names' in instances:
                        label_text = instances.label_names[i]
                    else:
                        label_text = classes[
                            label] if classes is not None else f'class {label}'
                    if 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        label_text += f': {score}'

        return self.get_image()
    
    def _draw_instances_bbox(self, image: np.ndarray, instances: ['InstanceData'],
                        color: Optional[tuple]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances and instances.bboxes.sum() > 0:
            bboxes = instances.bboxes

            colors = [color for bbox in bboxes]

            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=1)

        return self.get_image()
    
    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None
        result_img_data = None

        true_positive_color = tuple([0, 200, 0])
        false_positive_color = tuple([200, 0, 0])
        false_negative_color = tuple([0, 0, 200])

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and draw_pred and data_sample is not None:
            result_img_data = image
            if 'pred_instances' in data_sample:
                # Pred and gt
                if 'gt_instances' in data_sample:
                    true_positive_instances = InstanceData().new()
                    false_positive_instances = InstanceData().new()
                    false_negative_instances = InstanceData().new()
                    true_positive_instances_bboxes = []
                    false_positive_instances_bboxes = []
                    false_negative_instances_bboxes = []
                    pred_instances = data_sample.pred_instances
                    pred_instances = pred_instances[
                        pred_instances.scores > pred_score_thr]
                    gt_instances = data_sample.gt_instances
                    for pred_bbox in pred_instances.bboxes:
                        flag = False
                        for gt_bbox in gt_instances.bboxes:
                            if bbox_iou(pred_bbox, gt_bbox) > 0.5:
                                true_positive_instances_bboxes.append(pred_bbox.tolist())
                                flag = True
                                break
                        if flag == False:
                            false_positive_instances_bboxes.append(pred_bbox.tolist())

                    for gt_bbox in gt_instances.bboxes:
                        flag = False
                        for pred_bbox in pred_instances.bboxes:
                            if bbox_iou(pred_bbox, gt_bbox) > 0.35:
                                flag = True
                                break
                        if flag == False:
                            false_negative_instances_bboxes.append(gt_bbox.tolist())

                    true_positive_instances.bboxes = torch.Tensor(true_positive_instances_bboxes)
                    false_positive_instances.bboxes = torch.Tensor(false_positive_instances_bboxes)
                    false_negative_instances.bboxes = torch.Tensor(false_negative_instances_bboxes)
                    result_img_data = self._draw_instances_bbox(image, true_positive_instances,
                                                        true_positive_color)
                    result_img_data = self._draw_instances_bbox(result_img_data, false_positive_instances,
                                                        false_positive_color)
                    result_img_data = self._draw_instances_bbox(result_img_data, false_negative_instances,
                                                            false_negative_color)
                # False positive, only pred
                else:
                    pred_instances = data_sample.pred_instances
                    pred_instances = pred_instances[
                        pred_instances.scores > pred_score_thr]
                    result_img_data = self._draw_instances_bbox(image, pred_instances,
                                                        false_positive_color)
            # False negative, only gt
            elif 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances_bbox(image,
                                                   data_sample.gt_instances,
                                                   false_negative_color)

        elif draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)
            if 'gt_sem_seg' in data_sample:
                gt_img_data = self._draw_sem_seg(gt_img_data,
                                                 data_sample.gt_sem_seg,
                                                 classes, palette)

            if 'gt_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_panoptic_seg(
                    gt_img_data, data_sample.gt_panoptic_seg, classes, palette)

        elif draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]
                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette)

            if 'pred_sem_seg' in data_sample:
                pred_img_data = self._draw_sem_seg(pred_img_data,
                                                   data_sample.pred_sem_seg,
                                                   classes, palette)

            if 'pred_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_panoptic_seg(
                    pred_img_data, data_sample.pred_panoptic_seg.numpy(),
                    classes, palette)

        if result_img_data is not None:
            drawn_img = result_img_data
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)