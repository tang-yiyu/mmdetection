import warnings
import random
from typing import Optional

import mmengine
import mmengine.fileio as fileio
import mmcv
import numpy as np

from mmcv.transforms import LoadImageFromFile
from mmcv.transforms import to_tensor

from mmdet.registry import TRANSFORMS

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData, PixelData
from mmdet.structures.bbox import BaseBoxes, autocast_box_type

from mmdet.datasets.transforms.transforms import Resize, RandomFlip, RandomCrop, RandomErasing
from mmdet.datasets.transforms.formatting import PackDetInputs as PackDetInputs



Number = Union[int, float]


@TRANSFORMS.register_module()
class LoadTwoStreamImageFromFiles(LoadImageFromFile):
    """
    Load an image from file.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        filename_rgb = filename
        if 'rgb' in filename_rgb:
            filename_ir = filename_rgb.replace('rgb', 'ir')
        elif 'vis' in filename_rgb:
            filename_ir = filename_rgb.replace('vis', 'ir')
        elif 'VIS' in filename_rgb:
            filename_ir = filename_rgb.replace('VIS', 'IR')
        elif 'visible' in filename_rgb:
            filename_ir = filename_rgb.replace('visible', 'infrared')
        else:
            raise ValueError('Unknown filename format.')
        try:
            if self.file_client_args is not None:
                file_client_rgb = fileio.FileClient.infer_client(
                    self.file_client_args, filename_rgb)
                file_client_ir = fileio.FileClient.infer_client(
                    self.file_client_args, filename_ir)
                img_bytes_rgb = file_client_rgb.get(filename_rgb)
                img_bytes_ir = file_client_ir.get(filename_ir)
            else:
                img_bytes_rgb = fileio.get(
                    filename_rgb, backend_args=self.backend_args)
                img_bytes_ir = fileio.get(
                    filename_ir, backend_args=self.backend_args)
            img_rgb = mmcv.imfrombytes(
                img_bytes_rgb, flag=self.color_type, backend=self.imdecode_backend)
            img_ir = mmcv.imfrombytes(
                img_bytes_ir, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img_rgb is not None, f'failed to load visible image: {filename_rgb}'
        assert img_ir is not None, f'failed to load infrared image: {filename_ir}'
        if self.to_float32:
            img_rgb = img_rgb.astype(np.float32)
            img_ir = img_ir.astype(np.float32)

        results['img_rgb'] = img_rgb
        results['img_ir'] = img_ir
        assert img_rgb.shape[:2] == img_ir.shape[:2]
        results['img_shape'] = img_rgb.shape[:2]
        results['ori_shape'] = img_rgb.shape[:2]

        return results


@TRANSFORMS.register_module()
class ResizeTwoStream(Resize):
    """
    Resize images & bbox & seg.
    """

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if (results.get('img_rgb', None) is not None) and (results.get('img_ir', None) is not None):
            if self.keep_ratio:
                img_rgb, scale_factor = mmcv.imrescale(
                    results['img_rgb'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                img_ir, _ = mmcv.imrescale(
                    results['img_ir'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                assert img_rgb.shape[:2] == img_ir.shape[:2]
                new_h, new_w = img_rgb.shape[:2]
                
                assert results['img_rgb'].shape[:2] == results['img_ir'].shape[:2]   
                h, w = results['img_rgb'].shape[:2]
                
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img_rgb, w_scale, h_scale = mmcv.imresize(
                    results['img_rgb'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                img_ir, _ = mmcv.imresize(
                    results['img_ir'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img_rgb'] = img_rgb
            results['img_ir'] = img_ir
            assert img_rgb.shape[:2] == img_ir.shape[:2]
            results['img_shape'] = img_rgb.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio


@TRANSFORMS.register_module()
class RandomFlipTwoStream(RandomFlip):
    """
    Flip the image & bbox & mask & segmentation map. 
    """

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the RandomFlip."""
        cur_dir = results['flip_direction']
        assert results['img_rgb'].shape[:2] == results['img_ir'].shape[:2]
        h, w = results['img_rgb'].shape[:2]

        if cur_dir == 'horizontal':
            homography_matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'vertical':
            homography_matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'diagonal':
            homography_matrix = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        else:
            homography_matrix = np.eye(3, dtype=np.float32)

        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img_rgb'] = mmcv.imflip(
            results['img_rgb'], direction=results['flip_direction'])
        results['img_ir'] = mmcv.imflip(
            results['img_ir'], direction=results['flip_direction'])
        
        assert results['img_rgb'].shape[:2] == results['img_ir'].shape[:2]
        img_shape = results['img_rgb'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)


@TRANSFORMS.register_module()
class RandomCropTwoStream(RandomCrop):
    """
    Random crop the image & bboxes & masks.
    """

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img_rgb = results['img_rgb']
        img_ir = results['img_ir']
        assert img_rgb.shape[0] == img_ir.shape[0] and img_rgb.shape[1] == img_ir.shape[1]
        margin_h = max(img_rgb.shape[0] - crop_size[0], 0)
        margin_w = max(img_rgb.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img_rgb = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_ir = img_ir[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        assert img_rgb.shape == img_ir.shape
        img_shape = img_rgb.shape
        results['img_rgb'] = img_rgb
        results['img_ir'] = img_ir
        results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get('gt_instances_ids', None) is not None:
                results['gt_instances_ids'] = \
                    results['gt_instances_ids'][valid_inds]

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results

    
    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """

        assert results['img_rgb'].shape[:2] == results['img_ir'].shape[:2]
        image_size = results['img_rgb'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results
    

@TRANSFORMS.register_module()
class RandomErasingTwoStream(RandomErasing):
    """
    RandomErasing operation.
    """

    def _transform_img(self, results: dict, patches: List[list]) -> None:
        """Random erasing the image."""
        for patch in patches:
            px1, py1, px2, py2 = patch
            results['img_rgb'][py1:py2, px1:px2, :] = self.img_border_value
            results['img_ir'][py1:py2, px1:px2, :] = self.img_border_value



@TRANSFORMS.register_module()
class PackDetInputsTwoStream(PackDetInputs):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.
    """

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs_rgb' (obj:`torch.Tensor`): The forward data of models.
            - 'inputs_ir' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img_rgb' in results:
            img_rgb = results['img_rgb']
            if len(img_rgb.shape) < 3:
                img_rgb = np.expand_dims(img_rgb, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img_rgb.flags.c_contiguous:
                img_rgb = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
                img_rgb = to_tensor(img_rgb)
            else:
                img_rgb = to_tensor(img_rgb).permute(2, 0, 1).contiguous()

        if 'img_ir' in results:
            img_ir = results['img_ir']
            if len(img_ir.shape) < 3:
                img_ir = np.expand_dims(img_ir, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img_ir.flags.c_contiguous:
                img_ir = np.ascontiguousarray(img_ir.transpose(2, 0, 1))
                img_ir = to_tensor(img_ir)
            else:
                img_ir = to_tensor(img_ir).permute(2, 0, 1).contiguous()    

            # img = np.concatenate((img_rgb, img_ir), axis=0)
            # packed_results['inputs'] = img
            packed_results['inputs_rgb'] = img_rgb
            packed_results['inputs_ir'] = img_ir

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results


            
    