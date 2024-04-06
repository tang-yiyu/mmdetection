import os.path as osp
import mmcv

from typing import Sequence

from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.utils import mkdir_or_exist
from mmengine.fileio import get

from mmdet.registry import HOOKS
from mmdet.engine.hooks.visualization_hook import DetVisualizationHook
from mmdet.structures import DetDataSample


@HOOKS.register_module()
class TwoChannelDetVisualizationHook(DetVisualizationHook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path_rgb = outputs[0].img_path
        if 'rgb' in img_path_rgb:
            img_path_ir = img_path_rgb.replace('rgb', 'ir')
        elif 'vis' in img_path_rgb:
            img_path_ir = img_path_rgb.replace('vis', 'ir')
        elif 'VIS' in img_path_rgb:
            img_path_ir = img_path_rgb.replace('VIS', 'IR')
        elif 'visible' in img_path_rgb:
            img_path_ir = img_path_rgb.replace('visible', 'infrared')
        else:
            raise ValueError('Unknown img_path format.')
        
        # rgb
        img_bytes_rgb = get(img_path_rgb, backend_args=self.backend_args)
        img_rgb = mmcv.imfrombytes(img_bytes_rgb, channel_order='rgb')

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                osp.basename(img_path_rgb) if self.show else 'val_img',
                img_rgb,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)
        
        # ir
        img_bytes_ir = get(img_path_ir, backend_args=self.backend_args)
        img_ir = mmcv.imfrombytes(img_bytes_ir, channel_order='rgb')

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                osp.basename(img_path_ir) if self.show else 'val_img',
                img_ir,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        for data_sample in outputs:
            self._test_index += 1

            img_path_rgb = data_sample.img_path
            if 'rgb' in img_path_rgb:
                img_path_ir = img_path_rgb.replace('rgb', 'ir')
            elif 'vis' in img_path_rgb:
                img_path_ir = img_path_rgb.replace('vis', 'ir')
            elif 'VIS' in img_path_rgb:
                img_path_ir = img_path_rgb.replace('VIS', 'IR')
            elif 'visible' in img_path_rgb:
                img_path_ir = img_path_rgb.replace('visible', 'infrared')
            else:
                raise ValueError('Unknown img_path format.')

            # rgb
            img_bytes_rgb = get(img_path_rgb, backend_args=self.backend_args)
            img_rgb = mmcv.imfrombytes(img_bytes_rgb, channel_order='rgb')

            out_file_rgb = None
            if self.test_out_dir is not None:
                out_file_rgb = osp.basename(img_path_rgb.split('.')[0] + '_rgb.' + img_path_rgb.split('.')[-1])
                out_file_rgb = osp.join(self.test_out_dir, out_file_rgb)

            self._visualizer.add_datasample(
                osp.basename(img_path_rgb) if self.show else 'test_img',
                img_rgb,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file_rgb,
                step=self._test_index)
            
            # ir
            img_bytes_ir = get(img_path_ir, backend_args=self.backend_args)
            img_ir = mmcv.imfrombytes(img_bytes_ir, channel_order='rgb')

            out_file_ir = None
            if self.test_out_dir is not None:
                out_file_ir = osp.basename(img_path_ir.split('.')[0] + '_ir.' + img_path_ir.split('.')[-1])
                out_file_ir = osp.join(self.test_out_dir, out_file_ir)

            self._visualizer.add_datasample(
                osp.basename(img_path_ir) if self.show else 'test_img',
                img_ir,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file_ir,
                step=self._test_index)

