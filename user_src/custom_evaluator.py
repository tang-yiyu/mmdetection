import os
import torch
import numpy as np
import tempfile
import os.path as osp
import itertools
import datetime
import time
from pycocotools import mask as maskUtils
import copy

from collections import OrderedDict, defaultdict
from terminaltables import AsciiTable

from typing import Dict, List, Optional, Sequence, Union

from mmdet.registry import METRICS
from mmdet.evaluation.metrics.coco_metric import CocoMetric as CocoMetric
from mmdet.structures.mask import encode_mask_results
from mmdet.datasets.api_wrappers import COCO, COCOevalMP

from mmengine.logging import MMLogger
from mmengine.fileio import get_local_path, load


@METRICS.register_module()
class CocoMetricMod(CocoMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
        use_mp_eval (bool): Whether to use mul-processing evaluation
    """

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

            # save txt files
            result['img_path'] = data_sample['img_path']
            self.save_txt(result, self.outfile_prefix)

            
    def save_txt(self, result, outfile_prefix):
        label_file = result['img_path'].split('/')[-1].split('.')[0] + '.txt'
        result_path = os.path.join(outfile_prefix, 'labels')       
        if not os.path.exists(result_path):
            os.makedirs(result_path)       
        with open(os.path.join(result_path, label_file), 'w') as f:
            for i, bbox in enumerate(result['bboxes']):
                line = ('person', int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(result['scores'][i]))
                f.write(("%s " + "%g " * (len(line)-1)).rstrip() % line + "\n")


class COCOevaltiny:
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm',
                 ignore_uncertain=False, use_iod_for_ignore=False, use_ignore_attr=True,
                 use_iod_for_crowd=False):   # add by G
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Paramstiny(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
        # #### add by G ##########################################################################
        self.use_ignore_attr = use_ignore_attr
        self.use_iod_for_ignore = use_iod_for_ignore
        self.ignore_uncertain = ignore_uncertain
        self.use_iod_for_crowd = use_iod_for_crowd
        # ##########################################################################################

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            if self.use_ignore_attr:
                gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
                gt['ignore'] = ('iscrowd' in gt and gt['iscrowd']) or gt['ignore']  # changed by hui
            else:
                gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']

            # ########################################################### change by G ###############################
            if self.ignore_uncertain and 'uncertain' in gt and gt['uncertain']:
                gt['ignore'] = 1
            # ########################################################### change by G ###############################
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']


        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            if self.use_iod_for_crowd:
                computeIoU = self.mycomputeIoU
            else:
                computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def mycomputeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]

        ious = maskUtils.iou(d,g,iscrowd)

        # is_crowd_cat_g = [g if g['category_id']==2 for g in gt]
        # is_crowd_cat_d = [d if d['category_id']==2 for d in dt]
        is_crowd_cat_g = []
        is_crowd_cat_d = []
        is_crowd_cat_g_id = []
        is_crowd_cat_d_id = []
        for i, g in enumerate(gt):
            if g['category_id']==2:
                is_crowd_cat_g.append(g['bbox'])
                is_crowd_cat_g_id.append(i)
        for i, d in enumerate(dt):
            if d['category_id']==2:
                is_crowd_cat_d.append(d['bbox'])
                is_crowd_cat_d_id.append(i)
        if len(is_crowd_cat_d) > 0 and len(is_crowd_cat_g) > 0:
            is_crowd_cat_d = np.array(is_crowd_cat_d)
            is_crowd_cat_g = np.array(is_crowd_cat_g)

            iods = self.IOD(is_crowd_cat_d, is_crowd_cat_g)

            for id1, i in enumerate(is_crowd_cat_d_id):
                for id2, j in enumerate(is_crowd_cat_g_id):
                    ious[i, j] = iods[id1, id2]
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    # ###### add by G
    def IOD(self, dets, ignore_gts):
        def insect_boxes(box1, boxes):
            sx1, sy1, sx2, sy2 = box1[:4]
            tx1, ty1, tx2, ty2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            ix1 = np.where(tx1 > sx1, tx1, sx1)
            iy1 = np.where(ty1 > sy1, ty1, sy1)
            ix2 = np.where(tx2 < sx2, tx2, sx2)
            iy2 = np.where(ty2 < sy2, ty2, sy2)
            return np.array([ix1, iy1, ix2, iy2]).transpose((1, 0))

        def bbox_area(boxes):
            s = np.zeros(shape=(boxes.shape[0],), dtype=np.float32)
            tx1, ty1, tx2, ty2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            h = (tx2 - tx1)
            w = (ty2 - ty1)
            valid = np.all(np.array([h > 0, w > 0]), axis=0)
            s[valid] = (h * w)[valid]
            return s

        def bbox_iod(dets, gts, eps=1e-12):
            iods = np.zeros(shape=(dets.shape[0], gts.shape[0]), dtype=np.float32)
            dareas = bbox_area(dets)
            for i, (darea, det) in enumerate(zip(dareas, dets)):
                idet = insect_boxes(det, gts)
                iarea = bbox_area(idet)
                iods[i, :] = iarea / (darea + eps)
            return iods

        def xywh2xyxy(boxes):
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            return boxes
        from copy import deepcopy
        return bbox_iod(xywh2xyxy(deepcopy(dets)), xywh2xyxy(deepcopy(ignore_gts)))
        # return bbox_iod(xywh2xyxy(dets), xywh2xyxy(ignore_gts))

    def IOD_by_IOU(self, dets, ignore_gts, ignore_gts_area, ious):
        if ignore_gts_area is None:
            ignore_gts_area = ignore_gts[:, 2] * dets[:, 3]
        dets_area = dets[:, 2] * dets[:, 3]
        tile_dets_area = np.tile(dets_area.reshape((-1, 1)), (1, len(ignore_gts_area)))
        tile_gts_area = np.tile(ignore_gts_area.reshape((1, -1)), (len(dets_area), 1))
        iods = ious / (1 + ious) * (1 + tile_gts_area / tile_dets_area)
        return iods
    ########

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]   #gt为一个list保存了当前
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]

        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        # #### ad by G ##############
        ignore_gts = np.array([g['bbox'] for g in gt if g['_ignore']])
        ignore_gts_idx = np.array([i for i, g in enumerate(gt) if g['_ignore']])
        if len(ignore_gts_idx) > 0 and len(dt) > 0:
            ignore_gts_area = np.array([g['area'] for g in gt if g['_ignore']])  # use area
            ignore_ious = (ious.T[ignore_gts_idx]).T
        ######################
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        # #### ad by G ##############
                        if self.use_iod_for_ignore and len(ignore_gts) > 0:
                            # time from 156.88s -> 79. s
                            # iods = self.IOD_by_IOU(np.array([d['bbox']]), None, ignore_gts_area,
                            #                        ignore_ious[dind:dind+1, :])[0]
                            iods =self.IOD(np.array([d['bbox']]), ignore_gts)[0]
                            idx = np.argmax(iods)
                            if iods[idx] >= iou:
                                # print('inside')
                                m = ignore_gts_idx[idx]

                                dtIg[tind, dind] = gtIg[m]
                                dtm[tind, dind] = gt[m]['id']
                                gtm[tind, m] = d['id']
                            else: continue
                        else: continue
                        ######################
                        ###continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )   #统计了gt中非ignore的数量
                    # print("class:{}, area :{}, maxDet: {}, {}, {}".format(k0, a0, maxDet, npig, len(gtIg)))
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def float_equal(a, b):
            return np.abs(a-b) < 1e-6

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.4f}'  # change by hui {:0.3f} to {:0.4f}
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(float_equal(iouThr, p.iouThrs))[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(float_equal(iouThr, p.iouThrs))[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        
        def _summarizeDets_tiny():
            stats = []
            for isap in [1, 0]:
                for iouTh in self.params.iouThrs:
                    for areaRng in self.params.areaRngLbl:
                        stats.append(_summarize(isap, iouThr=iouTh, areaRng=areaRng, maxDets=self.params.maxDets[-1]))
            return np.array(stats)
        def _summarizeDets():
            # stats = np.zeros((12,))
            # stats[0] = _summarize(1)
            # stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            # stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            n = len(self.params.areaRngLbl)-1
            stats = []
            stats.extend([_summarize(1),
                          _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.6, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.7, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.8, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.9, maxDets=self.params.maxDets[2])])
            for i in range(n):
                # stats.append(_summarize(1, iouThr=0.5, areaRng=self.params.areaRngLbl[i+1], maxDets=self.params.maxDets[2]))
                stats.append(_summarize(1, areaRng=self.params.areaRngLbl[i+1], maxDets=self.params.maxDets[2]))
            stats.extend([_summarize(0, maxDets=self.params.maxDets[0]),
                          _summarize(0, maxDets=self.params.maxDets[1]),
                          _summarize(0, maxDets=self.params.maxDets[2])])
            for i in range(n):
                stats.append(_summarize(0, areaRng=self.params.areaRngLbl[i+1], maxDets=self.params.maxDets[2]))
            return stats
        
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            if self.params.EVAL_STRANDARD.startswith('tiny'): summarize = _summarizeDets_tiny
            else: summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Paramstiny:
    '''
    Params for coco evaluation api
    '''
    EVAL_STRANDARD = 'tiny'
    def setDetParams(self):
        eval_standard = Paramstiny.EVAL_STRANDARD.lower()
        if eval_standard.startswith('tiny'):
            self.imgIds = []
            self.catIds = []
            # np.arange causes trouble.  the data point on arange is slightly larger than the true value
            if eval_standard == 'tiny': self.iouThrs = np.array([0.25, 0.5, 0.75])
            elif eval_standard == 'tiny_sanya17': self.iouThrs = np.array([0.3, 0.5, 0.75])
            else: raise ValueError("eval_standard is not right: {}, must be 'tiny' or 'tiny_sanya17'".format(eval_standard))
            self.recThrs = np.linspace(.0, 1.00, int((1.00 - .0) / .01) + 1, endpoint=True)
            self.maxDets = [200]
            self.areaRng = [[1 ** 2, 1e5 ** 2], [1 ** 2, 20 ** 2], [1 ** 2, 8 ** 2], [8 ** 2, 12 ** 2],
                            [12 ** 2, 20 ** 2], [20 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]
            # s = 4.11886287119646
            # self.areaRng = np.array(self.areaRng) * (s ** 2)
            self.areaRngLbl = ['all', 'tiny', 'tiny1', 'tiny2', 'tiny3', 'small', 'reasonable']
            self.useCats = 1
        elif eval_standard == 'coco':  # COCO standard
            self.imgIds = []
            self.catIds = []
            # np.arange causes trouble.  the data point on arange is slightly larger than the true value
            self.iouThrs = np.linspace(.5, 0.95, int((0.95 - .5) / .05) + 1, endpoint=True)
            self.recThrs = np.linspace(.0, 1.00, int((1.00 - .0) / .01) + 1, endpoint=True)
            self.maxDets = [1, 10, 100]
            self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
            self.areaRngLbl = ['all', 'small', 'medium', 'large']
            self.useCats = 1

            self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 10 ** 2], [10 ** 2, 32 ** 2],
                            [32 ** 2, 96 ** 2], [96 ** 2, 288 ** 2], [288 ** 2, 1e5 ** 2]]
            self.areaRngLbl = ['all', 'out_small', 'in_small', 'medium', 'in_large', 'out_large']

            self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 64 ** 2], [64 ** 2, 128 ** 2],
                            [128 ** 2, 256 ** 2], [256 ** 2, 1024 ** 2], [1024 ** 2, 1e5 ** 2]]
            self.areaRngLbl = ['all', 'smallest', 'fpn1', 'fpn2', 'fpn3', 'fpn4+5', 'largest']
        else:
            raise ValueError('EVAL_STRANDARD not valid.')

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

@METRICS.register_module()
class DronePerson(CocoMetric):
    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            # iou_thrs = np.linspace(
            #     .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            iou_thrs = np.linspace(
                .25, 0.75, int(np.round((0.75 - .25) / .25)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOevaltiny(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP_25': 0,
                'mAP_25_tiny': 1,
                'mAP_25_tiny1':2,
                'mAP_25_tiny2':3,
                'mAP_25_tiny3':4,
                'mAP_25_small':5,
                'mAP_25_reasonable':6,
                'mAP_50': 7,
                'mAP_50_tiny': 8,
                'mAP_50_tiny1':9,
                'mAP_50_tiny2':10,
                'mAP_50_tiny3':11,
                'mAP_50_small':12,
                'mAP_50_reasonable':13,
                'mAP_75': 14,
                'mAP_75_tiny': 15,
                'mAP_75_tiny1':16,
                'mAP_75_tiny2':17,
                'mAP_75_tiny3':18,
                'mAP_75_small':19,
                'mAP_75_reasonable':20,
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[0, :, idx, 0, -1] #AP25
                        # precision = precisions[:, :, idx, 0, -1] #AP
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP25'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP_75', 'mAP_25',
                        'mAP_50', 'mAP_50_tiny', 'mAP_50_tiny1', 'mAP_50_tiny2', 'mAP_50_tiny3', 'mAP_50_small'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = coco_eval.stats[:8]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.4f} {ap[1]:.4f} '
                            f'{ap[2]:.4f} {ap[3]:.4f} {ap[4]:.4f} '
                            f'{ap[5]:.4f} {ap[6]:.4f} {ap[7]:.4f} ')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
