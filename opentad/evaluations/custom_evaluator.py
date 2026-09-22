# import json
# import numpy as np
# import copy
# import os
# from collections import OrderedDict
# import time
# try:
#     import mmcv
# except ImportError:
#     mmcv = None
# os.environ['TZ'] = 'Asia/Shanghai'
# try:
#     time.tzset()
# except AttributeError:
#     # Windows 系统不支持 tzset，通常跳过
#     pass
#
# from .builder import EVALUATORS
# # 导入官方 mAP 和去重函数
# from .mAP import mAP, remove_duplicate_annotations
#
#
# # ==========================================
# # 辅助函数
# # ==========================================
# def compute_iou(pred, gt):
#     start_pred, end_pred = pred[0], pred[1]
#     start_gt, end_gt = gt[0], gt[1]
#     intersection = max(0, min(end_pred, end_gt) - max(start_pred, start_gt))
#     union = (end_pred - start_pred) + (end_gt - start_gt) - intersection
#     return intersection / union if union > 0 else 0
#
#
# def compute_sl_match(pred, gt, threshold):
#     s_p, e_p = pred[0], pred[1]
#     s_g, e_g = gt[0], gt[1]
#     dur_gt = e_g - s_g
#     if dur_gt <= 0: return False
#     diff_s = abs(s_p - s_g) / dur_gt
#     diff_e = abs(e_p - e_g) / dur_gt
#     return (diff_s <= threshold) and (diff_e <= threshold)
#
#
# def levenshtein_distance(seq1, seq2):
#     size_x = len(seq1) + 1
#     size_y = len(seq2) + 1
#     matrix = np.zeros((size_x, size_y))
#     for x in range(size_x): matrix[x, 0] = x
#     for y in range(size_y): matrix[0, y] = y
#     for x in range(1, size_x):
#         for y in range(1, size_y):
#             if seq1[x - 1] == seq2[y - 1]:
#                 matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1)
#             else:
#                 matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1)
#     return matrix[size_x - 1, size_y - 1]
#
#
# # ==========================================
# # 自定义评估器
# # ==========================================
#
# @EVALUATORS.register_module()
# class CustomBMNEvaluator:
#     def __init__(self,
#                  ground_truth_filename,
#                  subset='validation',
#                  tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
#                  sl_thresholds=[0.1, 0.2, 0.3],
#                  f1_thresholds=[0.3, 0.5],
#                  mof_fps=1.0,
#                  f1_topk=20,  # [建议] 默认值从 200 改为 20，避免过多 FP
#                  f1_score_threshold=0.1,  # [新增] 默认过滤掉 0.1 分以下的预测
#                  **kwargs):
#
#         self.ground_truth_filename = ground_truth_filename
#         self.subset = subset
#         self.tiou_thresholds = tiou_thresholds
#         self.sl_thresholds = sl_thresholds
#         self.f1_thresholds = f1_thresholds
#         self.mof_fps = mof_fps
#         self.f1_topk = f1_topk
#         self.f1_score_threshold = f1_score_threshold  # 保存阈值
#
#         self.prediction_filename = kwargs.get('prediction_filename', None)
#
#         # 加载 GT
#         self.gt_dict, self.class_map, self.idx_to_class = self._load_gt_and_map()
#         self.metrics = {}
#
#     def _load_gt_and_map(self):
#         with open(self.ground_truth_filename, 'r') as fobj:
#             data = json.load(fobj)
#
#         all_labels = set()
#         for vid, info in data['database'].items():
#             for ann in info['annotations']:
#                 all_labels.add(ann['label'])
#
#         sorted_labels = sorted(list(all_labels))
#         class_map = {name: idx for idx, name in enumerate(sorted_labels)}
#         idx_to_class = {idx: name for idx, name in enumerate(sorted_labels)}
#
#         gt_dict = {}
#         for vid, info in data['database'].items():
#             if self.subset != 'all' and info['subset'] != self.subset:
#                 continue
#
#             clean_annotations = remove_duplicate_annotations(info['annotations'])
#
#             segments = []
#             for ann in clean_annotations:
#                 segments.append({
#                     'segment': ann['segment'],
#                     'label': class_map[ann['label']]
#                 })
#             gt_dict[vid] = {
#                 'segments': segments,
#                 'duration': info.get('duration', 0)
#             }
#         return gt_dict, class_map, idx_to_class
#
#     def _calculate_custom_ap(self, preds, gts, thresh, mode='sl'):
#         if not gts or not preds: return 0.0
#         preds = sorted(preds, key=lambda x: x['score'], reverse=True)
#         tp = np.zeros(len(preds))
#         fp = np.zeros(len(preds))
#         gt_matched = [False] * len(gts)
#
#         for i, p in enumerate(preds):
#             best_match_idx = -1
#             for j, g in enumerate(gts):
#                 if p['label'] != g['label']: continue
#                 if gt_matched[j]: continue
#                 if mode == 'sl':
#                     if compute_sl_match(p['segment'], g['segment'], thresh):
#                         best_match_idx = j
#                         break
#             if best_match_idx >= 0:
#                 tp[i] = 1
#                 gt_matched[best_match_idx] = True
#             else:
#                 fp[i] = 1
#
#         tp_cumsum = np.cumsum(tp)
#         fp_cumsum = np.cumsum(fp)
#         num_gt = len(gts)
#         recall = tp_cumsum / num_gt
#         precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
#
#         ap = 0
#         for t in np.arange(0., 1.1, 0.1):
#             if np.sum(recall >= t) == 0:
#                 p = 0
#             else:
#                 p = np.max(precision[recall >= t])
#             ap += p / 11.
#         return ap
#
#     def evaluate(self, results=None, logger=None):
#         if results is None:
#             if isinstance(self.prediction_filename, dict):
#                 results = self.prediction_filename
#             elif isinstance(self.prediction_filename, str) and self.prediction_filename and os.path.exists(
#                     self.prediction_filename):
#                 if self.prediction_filename.endswith('.json'):
#                     with open(self.prediction_filename, 'r') as f:
#                         results = json.load(f)
#                 else:
#                     if mmcv is not None: results = mmcv.load(self.prediction_filename)
#             else:
#                 if logger: logger.warning("No results found.")
#                 return {}
#
#         if isinstance(results, dict) and 'results' in results:
#             results = results['results']
#
#         if not isinstance(results, dict):
#             return {}
#
#         metrics = OrderedDict()
#
#         # Part 1: Official mAP
#         results_for_official = {}
#         for vid, preds in results.items():
#             new_preds = []
#             for p in preds:
#                 new_p = p.copy()
#                 lbl = p['label']
#                 if isinstance(lbl, (int, np.integer)):
#                     if lbl in self.idx_to_class:
#                         new_p['label'] = self.idx_to_class[lbl]
#                     else:
#                         continue
#                 elif isinstance(lbl, str):
#                     new_p['label'] = lbl
#                 new_preds.append(new_p)
#             results_for_official[vid] = new_preds
#
#         try:
#             official_evaluator = mAP(
#                 ground_truth_filename=self.ground_truth_filename,
#                 prediction_filename={'results': results_for_official},
#                 subset=self.subset,
#                 tiou_thresholds=self.tiou_thresholds,
#                 thread=4
#             )
#             official_metrics = official_evaluator.evaluate()
#             if official_metrics:
#                 metrics.update(official_metrics)
#         except Exception as e:
#             if logger: logger.error(f"Error calling official mAP logic: {e}")
#
#         # Part 2: Custom Metrics
#         valid_preds = {}
#         for vid, preds in results.items():
#             if vid in self.gt_dict:
#                 clean_preds = []
#                 for p in preds:
#                     lbl = p['label']
#                     if isinstance(lbl, str):
#                         if lbl in self.class_map:
#                             label_idx = self.class_map[lbl]
#                         else:
#                             continue
#                     else:
#                         label_idx = int(lbl)
#                     clean_preds.append({'segment': p['segment'], 'label': label_idx, 'score': float(p['score'])})
#                 valid_preds[vid] = clean_preds
#
#         # SL-mAP
#         sl_map_list = []
#         for thresh in self.sl_thresholds:
#             class_aps = []
#             for cls_name, cls_idx in self.class_map.items():
#                 c_preds = [];
#                 c_gts = []
#                 for vid in self.gt_dict:
#                     c_gts.extend([g for g in self.gt_dict[vid]['segments'] if g['label'] == cls_idx])
#                     if vid in valid_preds:
#                         c_preds.extend([p for p in valid_preds[vid] if p['label'] == cls_idx])
#                 ap = self._calculate_custom_ap(c_preds, c_gts, thresh, mode='sl')
#                 class_aps.append(ap)
#             sl_mAP = np.mean(class_aps)
#             sl_map_list.append(sl_mAP)
#             metrics[f'SL-mAP@{thresh:.2f}'] = sl_mAP
#         metrics['SL-mAP@avg'] = np.mean(sl_map_list)
#
#         # === F1 Score (Updated) ===
#         for thresh in self.f1_thresholds:
#             tp, fp, fn = 0, 0, 0
#             for vid in self.gt_dict:
#                 gts = self.gt_dict[vid]['segments']
#                 preds = valid_preds.get(vid, [])
#
#                 # [关键修改] 1. 先过滤掉低置信度的预测
#                 preds_filtered = [p for p in preds if p['score'] >= self.f1_score_threshold]
#
#                 # [关键修改] 2. 排序并取 Top-K (现在的 Top-K 是在经过置信度过滤后的)
#                 preds_sorted = sorted(preds_filtered, key=lambda x: x['score'], reverse=True)[:self.f1_topk]
#
#                 gt_hit = [False] * len(gts)
#                 vid_tp = 0
#                 for p in preds_sorted:
#                     match = False
#                     for i, g in enumerate(gts):
#                         if not gt_hit[i] and p['label'] == g['label']:
#                             if compute_iou(p['segment'], g['segment']) >= thresh:
#                                 gt_hit[i] = True;
#                                 match = True;
#                                 vid_tp += 1
#                                 break
#                     if not match: fp += 1
#                 tp += vid_tp;
#                 fn += len(gts) - vid_tp
#
#             precision = tp / (tp + fp + 1e-6)
#             recall = tp / (tp + fn + 1e-6)
#             f1 = 2 * precision * recall / (precision + recall + 1e-6)
#             metrics[f'F1@{thresh:.1f}'] = f1
#
#         # MOF & Edit Score
#         total_edit = 0;
#         mof_correct = 0;
#         mof_total = 0;
#         video_count = 0
#         for vid, gt_info in self.gt_dict.items():
#             duration = gt_info['duration']
#             if duration <= 0: continue
#             num_frames = int(duration * self.mof_fps)
#             if num_frames == 0: continue
#
#             gt_arr = np.full(num_frames, -1, dtype=int)
#             pred_arr = np.full(num_frames, -1, dtype=int)
#
#             for g in gt_info['segments']:
#                 s = max(0, int(g['segment'][0] * self.mof_fps))
#                 e = min(num_frames, int(g['segment'][1] * self.mof_fps))
#                 gt_arr[s:e] = g['label']
#
#             if vid in valid_preds:
#                 # MOF 也建议过滤一下低分的，否则全画上去会很乱，这里暂且保持只画最高分覆盖
#                 sorted_preds = sorted(valid_preds[vid], key=lambda x: x['score'])
#                 for p in sorted_preds:
#                     if p['score'] < 0.05: continue
#                     s = max(0, int(p['segment'][0] * self.mof_fps))
#                     e = min(num_frames, int(p['segment'][1] * self.mof_fps))
#                     pred_arr[s:e] = p['label']
#
#             mask = gt_arr != -1
#             if mask.sum() > 0:
#                 mof_correct += (gt_arr[mask] == pred_arr[mask]).sum()
#                 mof_total += mask.sum()
#
#             def get_seq(arr):
#                 seq = [];
#                 prev = -2
#                 for x in arr:
#                     if x != -1 and x != prev:
#                         seq.append(x); prev = x
#                     elif x == -1:
#                         prev = -1
#                 return seq
#
#             g_seq = get_seq(gt_arr);
#             p_seq = get_seq(pred_arr)
#             dist = levenshtein_distance(p_seq, g_seq)
#             max_len = max(len(p_seq), len(g_seq))
#             score = (1 - dist / max_len) * 100 if max_len > 0 else 100.0
#             total_edit += score;
#             video_count += 1
#
#         metrics['MOF'] = (mof_correct / mof_total * 100) if mof_total > 0 else 0.0
#         metrics['EditScore'] = total_edit / video_count if video_count > 0 else 0.0
#
#         self.metrics = metrics
#         return metrics
#
#     def logging(self, logger=None):
#         metrics = getattr(self, 'metrics', {})
#         log_str = "\n" + "-" * 20 + " Custom Evaluation " + "-" * 20 + "\n"
#         keys = list(metrics.keys())
#         map_keys = [k for k in keys if 'mAP' in k and 'SL' not in k]
#         other_keys = [k for k in keys if k not in map_keys]
#         for k in map_keys + other_keys:
#             v = metrics[k]
#             log_str += f"{k:<15}: {v:.4f}\n"
#         log_str += "-" * 60
#         if logger:
#             logger.info(log_str)
#         else:
#             print(log_str)
#


import json
import numpy as np
import copy
import os
from collections import OrderedDict
import time
try:
    import mmcv
except ImportError:
    mmcv = None
os.environ['TZ'] = 'Asia/Shanghai'
try:
    time.tzset()
except AttributeError:
    # Windows 系统不支持 tzset，通常跳过
    pass


from .builder import EVALUATORS
# 1. 导入 mAP 和 Recall (用于计算 AR@AN)
from .mAP import mAP, remove_duplicate_annotations
from .recall import Recall

# ==========================================
# 辅助函数
# ==========================================
def compute_iou(pred, gt):
    start_pred, end_pred = pred[0], pred[1]
    start_gt, end_gt = gt[0], gt[1]
    intersection = max(0, min(end_pred, end_gt) - max(start_pred, start_gt))
    union = (end_pred - start_pred) + (end_gt - start_gt) - intersection
    return intersection / union if union > 0 else 0


def compute_sl_match(pred, gt, threshold):
    s_p, e_p = pred[0], pred[1]
    s_g, e_g = gt[0], gt[1]
    dur_gt = e_g - s_g
    if dur_gt <= 0: return False
    diff_s = abs(s_p - s_g) / dur_gt
    diff_e = abs(e_p - e_g) / dur_gt
    return (diff_s <= threshold) and (diff_e <= threshold)


def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x): matrix[x, 0] = x
    for y in range(size_y): matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                cost = 0
            else:
                cost = 1
            matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x, y - 1] + 1, matrix[x - 1, y - 1] + cost)
    return matrix[size_x - 1, size_y - 1]


# ==========================================
# 自定义评估器
# ==========================================

@EVALUATORS.register_module()
class CustomBMNEvaluator:
    def __init__(self,
                 ground_truth_filename,
                 subset='validation',
                 tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                 sl_thresholds=[0.1, 0.2, 0.3],
                 f1_thresholds=[0.3, 0.5],
                 mof_fps=1.0,
                 f1_topk=20,
                 f1_score_threshold=0.1,
                 recall_topk=[1, 10, 100],
                 **kwargs):

        self.ground_truth_filename = ground_truth_filename
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.sl_thresholds = sl_thresholds
        self.f1_thresholds = f1_thresholds
        self.mof_fps = mof_fps
        self.f1_topk = f1_topk
        self.f1_score_threshold = f1_score_threshold
        self.recall_topk = recall_topk

        self.prediction_filename = kwargs.get('prediction_filename', None)

        self.gt_dict, self.class_map, self.idx_to_class = self._load_gt_and_map()
        self.metrics = {}

    def _load_gt_and_map(self):
        with open(self.ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)

        all_labels = set()
        for vid, info in data['database'].items():
            for ann in info['annotations']:
                all_labels.add(ann['label'])

        sorted_labels = sorted(list(all_labels))
        class_map = {name: idx for idx, name in enumerate(sorted_labels)}
        idx_to_class = {idx: name for idx, name in enumerate(sorted_labels)}

        gt_dict = {}
        for vid, info in data['database'].items():
            if self.subset != 'all' and info['subset'] != self.subset:
                continue

            clean_annotations = remove_duplicate_annotations(info['annotations'])
            segments = []
            for ann in clean_annotations:
                segments.append({
                    'segment': ann['segment'],
                    'label': class_map[ann['label']]
                })
            gt_dict[vid] = {
                'segments': segments,
                'duration': info.get('duration', 0)
            }
        return gt_dict, class_map, idx_to_class

    def _calculate_custom_ap(self, preds, gts, thresh, mode='sl'):
        if not gts or not preds: return 0.0
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_matched = [False] * len(gts)

        for i, p in enumerate(preds):
            best_match_idx = -1
            for j, g in enumerate(gts):
                if p['label'] != g['label']: continue
                if gt_matched[j]: continue
                if mode == 'sl':
                    if compute_sl_match(p['segment'], g['segment'], thresh):
                        best_match_idx = j
                        break
            if best_match_idx >= 0:
                tp[i] = 1
                gt_matched[best_match_idx] = True
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / len(gts)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        ap = 0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.
        return ap

    def evaluate(self, results=None, logger=None):
        # 1. 加载 Results
        if results is None:
            if isinstance(self.prediction_filename, dict):
                results = self.prediction_filename
            elif isinstance(self.prediction_filename, str) and self.prediction_filename and os.path.exists(
                    self.prediction_filename):
                if self.prediction_filename.endswith('.json'):
                    with open(self.prediction_filename, 'r') as f:
                        results = json.load(f)
                else:
                    if mmcv is not None: results = mmcv.load(self.prediction_filename)
            else:
                if logger: logger.warning("No results found.")
                return {}

        if isinstance(results, dict) and 'results' in results:
            results = results['results']

        if not isinstance(results, dict): return {}

        metrics = OrderedDict()

        # -----------------------------------------------------------
        # 准备数据：String Label (用于 mAP 和 Recall)
        # -----------------------------------------------------------
        results_for_official = {}
        for vid, preds in results.items():
            new_preds = []
            for p in preds:
                new_p = p.copy()
                lbl = p['label']
                if isinstance(lbl, (int, np.integer)):
                    if lbl in self.idx_to_class:
                        new_p['label'] = self.idx_to_class[lbl]
                    else:
                        continue
                elif isinstance(lbl, str):
                    new_p['label'] = lbl
                new_preds.append(new_p)
            results_for_official[vid] = new_preds

        official_input_dict = {'results': results_for_official}

        # =======================================================
        # Part 1: Official mAP (AP)
        # =======================================================
        try:
            official_evaluator = mAP(
                ground_truth_filename=self.ground_truth_filename,
                prediction_filename=official_input_dict,
                subset=self.subset,
                tiou_thresholds=self.tiou_thresholds,
                thread=4
            )
            map_metrics = official_evaluator.evaluate()
            if map_metrics: metrics.update(map_metrics)
        except Exception as e:
            # 打印详细错误以便调试
            print(f"[ERROR] mAP calculation failed: {e}")
            if logger: logger.error(f"Error calling mAP logic: {e}")

        # =======================================================
        # Part 2: Official Recall (AR@AN) [核心修复]
        # =======================================================
        try:
            recall_evaluator = Recall(
                ground_truth_filename=self.ground_truth_filename,
                prediction_filename=official_input_dict,
                subset=self.subset,
                tiou_thresholds=self.tiou_thresholds,
                topk=self.recall_topk,
                max_avg_nr_proposals=100
            )

            # [修复] 你的 Recall 类返回的是 dict，直接 update 即可
            recall_metrics = recall_evaluator.evaluate()

            if recall_metrics:
                # 你的 Recall 类返回的已经是 0-1 之间的小数，不需要乘 100
                # 因为你的 mAP 也是 0.94 这种格式，保持一致
                metrics.update(recall_metrics)

        except Exception as e:
            # 打印详细错误以便调试
            print(f"[ERROR] Recall calculation failed: {e}")
            if logger: logger.error(f"Error calling Recall logic: {e}")

        # =======================================================
        # Part 3: Custom Metrics (SL-mAP, F1, MOF, Edit)
        # =======================================================
        # 准备数据：Int Label
        valid_preds = {}
        for vid, preds in results.items():
            if vid in self.gt_dict:
                clean_preds = []
                for p in preds:
                    lbl = p['label']
                    if isinstance(lbl, str):
                        if lbl in self.class_map:
                            label_idx = self.class_map[lbl]
                        else:
                            continue
                    else:
                        label_idx = int(lbl)
                    clean_preds.append({'segment': p['segment'], 'label': label_idx, 'score': float(p['score'])})
                valid_preds[vid] = clean_preds

        # SL-mAP
        sl_map_list = []
        for thresh in self.sl_thresholds:
            class_aps = []
            for cls_name, cls_idx in self.class_map.items():
                c_preds = [];
                c_gts = []
                for vid in self.gt_dict:
                    c_gts.extend([g for g in self.gt_dict[vid]['segments'] if g['label'] == cls_idx])
                    if vid in valid_preds:
                        c_preds.extend([p for p in valid_preds[vid] if p['label'] == cls_idx])
                ap = self._calculate_custom_ap(c_preds, c_gts, thresh, mode='sl')
                class_aps.append(ap)
            sl_mAP = np.mean(class_aps)
            sl_map_list.append(sl_mAP)
            metrics[f'SL-mAP@{thresh:.2f}'] = sl_mAP
        metrics['SL-mAP@avg'] = np.mean(sl_map_list)

        # F1 Score
        for thresh in self.f1_thresholds:
            tp, fp, fn = 0, 0, 0
            for vid in self.gt_dict:
                gts = self.gt_dict[vid]['segments']
                preds = valid_preds.get(vid, [])
                preds_filtered = [p for p in preds if p['score'] >= self.f1_score_threshold]
                preds_sorted = sorted(preds_filtered, key=lambda x: x['score'], reverse=True)[:self.f1_topk]

                gt_hit = [False] * len(gts)
                vid_tp = 0
                for p in preds_sorted:
                    match = False
                    for i, g in enumerate(gts):
                        if not gt_hit[i] and p['label'] == g['label']:
                            if compute_iou(p['segment'], g['segment']) >= thresh:
                                gt_hit[i] = True;
                                match = True;
                                vid_tp += 1
                                break
                    if not match: fp += 1
                tp += vid_tp;
                fn += len(gts) - vid_tp
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            metrics[f'F1@{thresh:.1f}'] = f1

        # MOF & Edit Score
        total_edit = 0;
        mof_correct = 0;
        mof_total = 0;
        video_count = 0
        for vid, gt_info in self.gt_dict.items():
            duration = gt_info['duration']
            if duration <= 0: continue
            num_frames = int(duration * self.mof_fps)
            if num_frames == 0: continue

            gt_arr = np.full(num_frames, -1, dtype=int)
            pred_arr = np.full(num_frames, -1, dtype=int)

            for g in gt_info['segments']:
                s = max(0, int(g['segment'][0] * self.mof_fps))
                e = min(num_frames, int(g['segment'][1] * self.mof_fps))
                gt_arr[s:e] = g['label']

            if vid in valid_preds:
                sorted_preds = sorted(valid_preds[vid], key=lambda x: x['score'])
                for p in sorted_preds:
                    if p['score'] < self.f1_score_threshold: continue
                    s = max(0, int(p['segment'][0] * self.mof_fps))
                    e = min(num_frames, int(p['segment'][1] * self.mof_fps))
                    pred_arr[s:e] = p['label']

            mask = np.ones_like(gt_arr, dtype=bool)
            if mask.sum() > 0:
                mof_correct += (gt_arr[mask] == pred_arr[mask]).sum()
                mof_total += mask.sum()

            def get_seq(arr):
                seq = [];
                prev = -2
                for x in arr:
                    if x != -1 and x != prev:
                        seq.append(x); prev = x
                    elif x == -1:
                        prev = -1
                return seq

            g_seq = get_seq(gt_arr);
            p_seq = get_seq(pred_arr)
            max_len = max(len(p_seq), len(g_seq))
            if max_len == 0:
                score = 100.0
            else:
                dist = levenshtein_distance(p_seq, g_seq)
                score = (1 - dist / max_len) * 100
            total_edit += score;
            video_count += 1

        metrics['MOF'] = (mof_correct / mof_total * 100) if mof_total > 0 else 0.0
        metrics['EditScore'] = total_edit / video_count if video_count > 0 else 0.0

        self.metrics = metrics
        return metrics

    def logging(self, logger=None):
        metrics = getattr(self, 'metrics', {})
        log_str = "\n" + "-" * 20 + " Custom Evaluation " + "-" * 20 + "\n"

        # 优化打印顺序
        keys = list(metrics.keys())
        groups = {'mAP': [], 'AR': [], 'SL': [], 'F1': [], 'Other': []}
        for k in keys:
            if k.startswith('mAP') or k == 'average_mAP':
                groups['mAP'].append(k)
            elif k.startswith('AR'):
                groups['AR'].append(k)
            elif k.startswith('SL'):
                groups['SL'].append(k)
            elif k.startswith('F1'):
                groups['F1'].append(k)
            else:
                groups['Other'].append(k)

        for g in ['mAP', 'AR', 'SL', 'F1', 'Other']:
            # 按名称排序，保证输出整齐
            for k in sorted(groups[g]):
                v = metrics[k]
                log_str += f"{k:<15}: {v:.4f}\n"

        log_str += "-" * 60
        if logger:
            logger.info(log_str)
        else:
            print(log_str)
