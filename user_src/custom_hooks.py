import os.path as osp
import mmcv

from typing import Sequence, Optional, Union

from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.utils import mkdir_or_exist
from mmengine.fileio import get
from mmengine.model.wrappers import is_model_wrapper

from mmdet.registry import HOOKS
from mmdet.engine.hooks.visualization_hook import DetVisualizationHook
from mmdet.structures import DetDataSample

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class TwoStreamDetVisualizationHook(DetVisualizationHook):

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
                # draw_pred=False,
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
                # draw_pred=False,
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
                # draw_pred=False,
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
                # draw_pred=False,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file_ir,
                step=self._test_index)


@HOOKS.register_module()
class AdjustModeHook(Hook):

    def freeze_policy_net(self):
        self.model.update_policy_net = False
        # for param in self.model.feature_rgb.parameters():
        #     param.requires_grad = False

        # for param in self.model.feature_ir.parameters():
        #     param.requires_grad = False

        # for param in self.model.policy_net.parameters():
        #     param.requires_grad = False

        for selection_layer in self.model.selection_layers:
            for param in selection_layer.parameters():
                param.requires_grad = False

        # for policy_layer in self.model.policy_layers:
        #     for param in policy_layer.parameters():
        #         param.requires_grad = False

    def unfreeze_policy_net(self):
        self.model.update_policy_net = True
        # for param in self.model.feature_rgb.parameters():
        #     param.requires_grad = True

        # for param in self.model.feature_ir.parameters():
        #     param.requires_grad = True

        # for param in self.model.policy_net.parameters():
        #     param.requires_grad = True

        for selection_layer in self.model.selection_layers:
            for param in selection_layer.parameters():
                param.requires_grad = True
            
        # for policy_layer in self.model.policy_layers:
        #     for param in policy_layer.parameters():
        #         param.requires_grad = True

    def freeze_main_net(self):
        self.model.update_main_net = False
        for param in self.model.backbone_rgb.parameters():
            param.requires_grad = False
        
        for param in self.model.backbone_ir.parameters():
            param.requires_grad = False

        for param in self.model.neck_rgb.parameters():
            param.requires_grad = False

        for param in self.model.neck_ir.parameters():
            param.requires_grad = False
        
        for param in self.model.fusion_layers.parameters():
            param.requires_grad = False

        for param in self.model.rpn_head.parameters():
            param.requires_grad = False

        for param in self.model.roi_head.parameters():
            param.requires_grad = False

    def unfreeze_main_net(self):
        self.model.update_main_net = True
        for param in self.model.backbone_rgb.parameters():
            param.requires_grad = True
        
        for param in self.model.backbone_ir.parameters():
            param.requires_grad = True

        for param in self.model.neck_rgb.parameters():
            param.requires_grad = True

        for param in self.model.neck_ir.parameters():
            param.requires_grad = True

        for param in self.model.fusion_layers.parameters():
            param.requires_grad = True

        for param in self.model.rpn_head.parameters():
            param.requires_grad = True

        for param in self.model.roi_head.parameters():
            param.requires_grad = True

    def before_train_epoch(self, runner) -> None:
        epoch = runner.epoch
        self.model = runner.model
        if is_model_wrapper(self.model):
            self.model = self.model.module

        warmup_epoch = 5
        prepare_epoch = warmup_epoch + 5
        alternate_epoch = runner.max_epochs - 15

        if self.every_n_epochs(runner, 1, 0):
            # Warm up training
            if epoch == 0 and epoch != warmup_epoch:
                print("Warmup Training")

            # Prepare training
            elif epoch >= warmup_epoch and epoch < prepare_epoch:
                if epoch == warmup_epoch:
                    print("Prepare Training")
                self.freeze_policy_net()
                self.unfreeze_main_net()
                # for k,v in self.model.named_parameters():
                #     print('{}: {}'.format(k, v.requires_grad))
                # print('Done!')

            # Alternate training
            elif prepare_epoch <= epoch < alternate_epoch:
                if epoch == prepare_epoch:
                    print("Alternate Training")
                if (epoch - prepare_epoch) % 4 == 0:
                    print("Update PolicyNet")
                    self.unfreeze_policy_net()
                    self.freeze_main_net()
                    if epoch > prepare_epoch:
                        # self.model.policy_net.decay_temperature(0.85)
                        for selection_layer in self.model.selection_layers:
                            selection_layer.decay_temperature(0.85)
                        # for policy_layer in self.model.policy_layers:
                        #     policy_layer.decay_temperature(0.85)
                elif ((epoch - prepare_epoch) % 2 == 0) and ((epoch - prepare_epoch) % 4 != 0):
                    print("Update MainNet")
                    self.freeze_policy_net()
                    self.unfreeze_main_net()
            
            # Finetune training
            elif epoch >= alternate_epoch:
                if epoch == alternate_epoch:
                    print("Finetune Training")
                self.freeze_policy_net()
                self.unfreeze_main_net()
            