"""
    Evaluation-related codes are modified from
    https://github.com/hughw19/NOCS_CVPR2019
"""
import logging
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
from tqdm import tqdm

# 这段代码定义了一个名为setup_logger的函数，用于设置Python日志记录系统，它创建了一个具有指定名称（logger_name）、日志级别（默认为INFO级别）和日志保存文件（log_file）的日志记录器。下面是对这个函数的详细解析：
#     获取或创建日志记录器：
#         logging.getLogger(logger_name)：这个方法根据提供的名称logger_name获取一个日志记录器（如果已存在则直接获取，否则新建一个）。日志记录器是日志系统中的核心组件，负责处理日志的生成和分发。
#     定义日志格式：
#         logging.Formatter('%(asctime)s : %(message)s')：创建一个日志格式化器，设置日志信息的显示样式。这里的格式化字符串表示每条日志将包含时间戳（%(asctime)s）和消息内容（%(message)s）。
#     添加文件处理器：
#         logging.FileHandler(log_file, mode='a')：创建一个文件处理器，指定日志将被写入到log_file指定的文件中，且文件打开模式为追加（'a'），即在文件末尾追加新的日志内容而不会覆盖原有内容。
#         fileHandler.setFormatter(formatter)：将之前定义的日志格式化器应用到文件处理器上，决定输出到文件的日志格式。
#     设置日志级别和添加文件处理器到日志记录器：
#         logger.setLevel(level)：设置日志记录器的最低日志级别为level，默认是INFO级别，这意味着只有INFO及以上级别的日志（如INFO, WARNING, ERROR, CRITICAL）会被处理。
#         logger.addHandler(fileHandler)：将文件处理器添加到日志记录器中，使得日志记录器产生的日志也会被输出到文件中。
#     添加屏幕输出处理器（控制台处理器）：
#         streamHandler = logging.StreamHandler()：创建一个控制台处理器，用于将日志输出到终端或控制台。
#         streamHandler.setFormatter(formatter)：同样设置其格式化器，使控制台输出的日志格式与文件中的一致。
#         logger.addHandler(streamHandler)：将控制台处理器添加到日志记录器，使得日志同时出现在屏幕和文件中。
#     返回日志记录器：
#         return logger：最后，函数返回配置好的日志记录器，供外部代码使用。
# 通过调用这个函数，你可以轻松地为你的应用程序或模块设置详细的日志记录功能，既可以在控制台上看到实时的日志输出，又能在指定文件中保存日志记录，便于后续分析和故障排查。
def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger


############################################################
#  Evaluation
############################################################

def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]
    """
    # 通过长宽高，计算立方体的8个定点，然后再将立方体的8个顶点的三个坐标分别与偏置量相加，那么就得到了立方体8个定点的坐标系。但是此时并没有计算旋转问题。
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]
    Returns:
        new_coordinates: [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def compute_3d_IoU(sRT_1, sRT_2, size_1, size_2, class_name_1, class_name_2, handle_visibility):
    """ Computes IoU overlaps between two 3D bboxes. """

    def asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2):
        noc_cube_1 = get_3d_bbox(size_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, sRT_1)
        noc_cube_2 = get_3d_bbox(size_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, sRT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    if sRT_1 is None or sRT_2 is None:
        return -1

    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or \
            (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility == 0):
        def y_rotation_matrix(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                             [0, 1, 0, 0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0, 0, 0, 1]])

        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = sRT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
            max_iou = max(max_iou, asymmetric_3d_iou(rotated_RT_1, sRT_2, size_1, size_2))
    else:
        max_iou = asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2)

    return max_iou


def compute_IoU_matches(gt_class_ids, gt_sRT, gt_size, gt_handle_visibility,
                        pred_class_ids, pred_sRT, pred_size, pred_scores,
                        synset_names, iou_3d_thresholds, score_threshold=0):
    """ Find matches between NOCS prediction and ground truth instances.
    Args:
        size: 3D bounding box size
        bboxes: 2D bounding boxes
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of the matched ground truth box.
        overlaps: IoU overlaps.
        indices:
    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)
    if num_pred:
        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        pred_class_ids = pred_class_ids[indices].copy()
        pred_size = pred_size[indices].copy()
        pred_sRT = pred_sRT[indices].copy()
    # compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_3d_IoU(pred_sRT[i], gt_sRT[j], pred_size[i, :], gt_size[j],
                                            synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]],
                                            gt_handle_visibility[j])
    # loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])
    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(indices.shape[0]):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue
                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break
    return gt_matches, pred_matches, overlaps, indices


def compute_RT_errors(sRT_1, sRT_2, class_id, handle_visibility, synset_names):
    """
    Args:
        sRT_1: [4, 4]. homogeneous affine transformation
        sRT_2: [4, 4]. homogeneous affine transformation
    Returns:
        theta: angle difference of R in degree
        shift: l2 difference of T in centimeter
    """
    # make sure the last row is [0, 0, 0, 1]
    if sRT_1 is None or sRT_2 is None:
        return -1
    try:
        assert np.array_equal(sRT_1[3, :], sRT_2[3, :])
        assert np.array_equal(sRT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT_1[3, :], sRT_2[3, :])
        exit()

    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    T1 = sRT_1[:3, 3]
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
    T2 = sRT_2[:3, 3]
    # symmetric when rotating around y-axis
    if synset_names[class_id] in ['bottle', 'can', 'bowl'] or \
            (synset_names[class_id] == 'mug' and handle_visibility == 0):
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
    else:
        R = R1 @ R2.transpose()
        cos_theta = (np.trace(R) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result


def compute_RT_overlaps(gt_class_ids, gt_sRT, gt_handle_visibility, pred_class_ids, pred_sRT, synset_names):
    """ Finds overlaps between prediction and ground truth instances.
    Returns:
        overlaps:
    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_errors(pred_sRT[i], gt_sRT[j], gt_class_ids[j],
                                                  gt_handle_visibility[j], synset_names)
    return overlaps


def compute_RT_matches(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue
                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches


def compute_ap_and_acc(pred_matches, pred_scores, gt_matches):
    # sort the scores from high to low
    assert pred_matches.shape[0] == pred_scores.shape[0]
    score_indices = np.argsort(pred_scores)[::-1]
    # pred_scores = pred_scores[score_indices]
    pred_matches = pred_matches[score_indices]
    precisions = np.cumsum(pred_matches > -1) / (np.arange(len(pred_matches)) + 1)
    recalls = np.cumsum(pred_matches > -1).astype(np.float32) / len(gt_matches)
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    # compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    # accuracy
    acc = np.sum(pred_matches > -1) / len(pred_matches)

    return ap, acc


def compute_mAP(pred_results, out_dir, degree_thresholds=[180], shift_thresholds=[100],
                iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False, cls_ids=None):
    """ Compute mean Average Precision.
    Returns:
        iou_aps:
        pose_aps:
        iou_acc:
        pose_acc:
    """
    synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)
    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)
    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)
    if cls_ids is None:
        cls_ids = range(1, num_classes)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    # pre-allocate more than enough memory
    iou_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_acc = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_count = [0 for _ in range(num_classes)]
    iou_gt_count = [0 for _ in range(num_classes)]

    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_acc = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_gt_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_count = [0 for _ in range(num_classes)]
    pose_gt_count = [0 for _ in range(num_classes)]

    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    for progress, result in enumerate(tqdm(pred_results)):
        gt_class_ids = np.array(result['gt_class_ids']).astype(np.int32)
        gt_RT = np.array(result['gt_RTs'])
        gt_size = np.array(result['gt_scales'])
        gt_handle_visibility = np.array(result['gt_handle_visibility'])

        pred_class_ids = np.array(result['pred_class_ids'])
        pred_sRT = np.array(result['pred_RTs'])
        pred_size = np.array(result['pred_scales'])
        pred_scores = np.array(result['pred_scores'])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in cls_ids:
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_sRT = gt_RT[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))
            cls_gt_size = gt_size[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)

            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_sRT = pred_sRT[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_size = pred_size[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)

            # calculate the overlap between each gt instance and pred instance
            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = \
                compute_IoU_matches(cls_gt_class_ids, cls_gt_sRT, cls_gt_size, cls_gt_handle_visibility,
                                    cls_pred_class_ids, cls_pred_sRT, cls_pred_size, cls_pred_scores,
                                    synset_names, iou_thres_list)
            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_sRT = cls_pred_sRT[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]

            num_pred = iou_cls_pred_match.shape[1]
            pred_start = iou_pred_count[cls_id]
            pred_end = pred_start + num_pred
            iou_pred_count[cls_id] = pred_end
            iou_pred_matches_all[cls_id][:, pred_start:pred_end] = iou_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            assert cls_pred_scores_tile.shape[1] == num_pred
            iou_pred_scores_all[cls_id][:, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = iou_cls_gt_match.shape[1]
            gt_start = iou_gt_count[cls_id]
            gt_end = gt_start + num_gt
            iou_gt_count[cls_id] = gt_end
            iou_gt_matches_all[cls_id][:, gt_start:gt_end] = iou_cls_gt_match

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)
                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_sRT = cls_pred_sRT[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_sRT = cls_gt_sRT[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)

            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
                                              cls_pred_class_ids, cls_pred_sRT, synset_names)
            pose_cls_gt_match, pose_cls_pred_match = compute_RT_matches(RT_overlaps, cls_pred_class_ids, cls_gt_class_ids,
                                                                        degree_thres_list, shift_thres_list)
            num_pred = pose_cls_pred_match.shape[2]
            pred_start = pose_pred_count[cls_id]
            pred_end = pred_start + num_pred
            pose_pred_count[cls_id] = pred_end
            pose_pred_matches_all[cls_id][:, :, pred_start:pred_end] = pose_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            assert cls_pred_scores_tile.shape[2] == num_pred
            pose_pred_scores_all[cls_id][:, :, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = pose_cls_gt_match.shape[2]
            gt_start = pose_gt_count[cls_id]
            gt_end = gt_start + num_gt
            pose_gt_count[cls_id] = gt_end
            pose_gt_matches_all[cls_id][:, :, gt_start:gt_end] = pose_cls_gt_match

    # trim zeros
    for cls_id in cls_ids:
        # IoU
        iou_pred_matches_all[cls_id] = iou_pred_matches_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_pred_scores_all[cls_id] = iou_pred_scores_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_gt_matches_all[cls_id] = iou_gt_matches_all[cls_id][:, :iou_gt_count[cls_id]]
        # pose
        pose_pred_matches_all[cls_id] = pose_pred_matches_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_pred_scores_all[cls_id] = pose_pred_scores_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_gt_matches_all[cls_id] = pose_gt_matches_all[cls_id][:, :, :pose_gt_count[cls_id]]

    # compute 3D IoU mAP
    for cls_id in range(1, num_classes):
        for s, iou_thres in enumerate(iou_thres_list):
            iou_aps[cls_id, s], iou_acc[cls_id, s] = compute_ap_and_acc(iou_pred_matches_all[cls_id][s, :],
                                                                        iou_pred_scores_all[cls_id][s, :],
                                                                        iou_gt_matches_all[cls_id][s, :])
    iou_aps[-1, :] = np.sum(iou_aps[1:-1, :], axis=0) / len(cls_ids)
    iou_acc[-1, :] = np.sum(iou_acc[1:-1, :], axis=0) / len(cls_ids)
    # compute pose mAP
    for i, degree_thres in enumerate(degree_thres_list):
        for j, shift_thres in enumerate(shift_thres_list):
            for cls_id in cls_ids:
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]
                pose_aps[cls_id, i, j], pose_acc[cls_id, i, j] = compute_ap_and_acc(cls_pose_pred_matches_all,
                                                                                    cls_pose_pred_scores_all,
                                                                                    cls_pose_gt_matches_all)
            pose_aps[-1, i, j] = np.sum(pose_aps[1:-1, i, j]) / len(cls_ids)
            pose_acc[-1, i, j] = np.sum(pose_acc[1:-1, i, j]) / len(cls_ids)

    # save results to pkl
    result_dict = {}
    result_dict['iou_thres_list'] = iou_thres_list
    result_dict['degree_thres_list'] = degree_thres_list
    result_dict['shift_thres_list'] = shift_thres_list
    result_dict['iou_aps'] = iou_aps
    result_dict['pose_aps'] = pose_aps
    result_dict['iou_acc'] = iou_acc
    result_dict['pose_acc'] = pose_acc
    pkl_path = os.path.join(out_dir, 'mAP_Acc.pkl')
    with open(pkl_path, 'wb') as f:
        cPickle.dump(result_dict, f)
    return iou_aps, pose_aps, iou_acc, pose_acc


def plot_mAP(iou_aps, pose_aps, out_dir, iou_thres_list, degree_thres_list, shift_thres_list):
    """ Draw iou 3d AP vs. iou thresholds.
    """

    labels = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug', 'mean', 'nocs']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:purple', 'tab:red', 'tab:gray']
    styles = ['-', '-', '-', '-', '-', '-', '--', ':']

    fig, (ax_iou, ax_degree, ax_shift) = plt.subplots(1, 3, figsize=(8, 3.5))
    # IoU subplot
    ax_iou.set_title('3D IoU', fontsize=10)
    ax_iou.set_ylabel('Average Precision')
    ax_iou.set_ylim(0, 100)
    ax_iou.set_xlabel('Percent')
    ax_iou.set_xlim(0, 100)
    ax_iou.xaxis.set_ticks([0, 25, 50, 75, 100])
    ax_iou.grid()
    for i in range(1, iou_aps.shape[0]):
        ax_iou.plot(100*np.array(iou_thres_list), 100*iou_aps[i, :],
                    color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # rotation subplot
    ax_degree.set_title('Rotation', fontsize=10)
    ax_degree.set_ylim(0, 100)
    ax_degree.yaxis.set_ticklabels([])
    ax_degree.set_xlabel('Degree')
    ax_degree.set_xlim(0, 60)
    ax_degree.xaxis.set_ticks([0, 20, 40, 60])
    ax_degree.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_degree.plot(np.array(degree_thres_list), 100 * pose_aps[i, :len(degree_thres_list), -1],
                       color=colors[i - 1], linestyle=styles[i - 1], label=labels[i - 1])
    # translation subplot
    ax_shift.set_title('Translation', fontsize=10)
    ax_shift.set_ylim(0, 100)
    ax_shift.yaxis.set_ticklabels([])
    ax_shift.set_xlabel('Centimeter')
    ax_shift.set_xlim(0, 10)
    ax_shift.xaxis.set_ticks([0, 5, 10])
    ax_shift.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_shift.plot(np.array(shift_thres_list), 100 * pose_aps[i, -1, :len(shift_thres_list)],
                      color=colors[i - 1], linestyle=styles[i - 1], label=labels[i - 1])
    ax_shift.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'mAP.png'))
    plt.close(fig)
    return


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]
    Returns:
        projected_coordinates: [N, 2]


    内参矩阵的一般形式如下（假设没有倾斜，即理想的针孔相机模型）：
    [
    \begin{bmatrix}
    f_x & 0 & c_x \
    0 & f_y & c_y \
    0 & 0 & 1
    \end{bmatrix}
    ]

    该函数 calculate_2d_projections 的目的是将三维空间中的点投影到二维平面上，这通常是三维重建、计算机视觉以及相机校准等领域中的一个基础步骤。
    它使用了摄像机的内参矩阵（intrinsics）来进行投影变换。以下是详细说明：
    参数解释
    coordinates_3d: 一个形状为 [3, N] 的NumPy数组，其中N表示空间中点的数量。每一列代表一个三维点的坐标(x, y, z)，第一行是所有点的x坐标，第二行是y坐标，第三行是z坐标。
    intrinsics: 一个形状为 [3, 3] 的NumPy数组，代表摄像机的内参矩阵。
    内参矩阵包含了焦距、主点坐标等信息，用于将三维空间中的点转换为二维图像平面上的像素坐标。
    函数执行流程
    矩阵乘法: 使用内参矩阵 intrinsics 与三维点坐标 coordinates_3d 进行矩阵乘法。这是将三维坐标变换到二维图像平面的关键步骤，乘法结果暂时包含了齐次坐标。
    齐次除法: 将乘法结果的前两行（即二维投影坐标）除以其第三行（齐次分量），这一步是为了从齐次坐标转换回笛卡尔坐标，完成透视除法，得到归一化的二维坐标。
    转置与类型转换: 随后，将得到的坐标矩阵转置，使其形式为 [N, 2]，即每行代表一个二维点的坐标。
    最后，将坐标转换为整型（np.int32），因为图像像素坐标通常为整数。
    返回值
        projected_coordinates: 一个形状为 [N, 2] 的NumPy数组，其中每一行代表原三维空间点投影到二维图像平面上的像素坐标。
    此函数的核心在于利用摄像机的内参矩阵完成从三维到二维的投影，是理解摄像机几何和进行三维重建不可或缺的一部分。
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def align_rotation(sRT):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
        该函数 align_rotation 的目的是对称对象的旋转矩阵进行对齐处理，确保旋转表示的一致性。它主要用于处理3D目标检测或姿态估计任务中，特别是在对象具有旋转对称性的情况下，需要规范旋转表示以避免多重解的问题。下面是对函数逻辑的详细解析：
输入参数
    sRT: 一个4x4的矩阵，表示一个刚体变换矩阵，包含了缩放(scaling)、旋转(rotation)和平移(translation)信息。其中，前3x3子矩阵代表旋转和缩放信息，最后一列为平移向量。
函数执行步骤
    计算缩放因子(s): 使用立方根计算3x3旋转矩阵行列式的值的立方根，得到缩放因子s。行列式的值实际上表示了体积缩放比例。
    提取旋转矩阵(R)和位移向量(T): 从sRT中分离出旋转矩阵R和位移向量T。
    计算简化旋转参数:
        计算两个旋转参数theta_x和theta_y，它们是从旋转矩阵转换来的，用于后续构建映射矩阵s_map。这两个参数帮助规范旋转表达，确保旋转表示的一致性。
        计算r_norm，即theta_x和theta_y的欧几里得范数，用于归一化。
    构建旋转对齐映射矩阵(s_map): 根据theta_x和theta_y计算出一个3x3的旋转对齐映射矩阵s_map，用于消除旋转表示的冗余，确保旋转矩阵的一致性，尤其是在处理具有旋转对称性的对象时。
    旋转矩阵对齐: 将原始旋转矩阵R与对齐映射矩阵s_map相乘，得到对齐后的旋转矩阵。
    组合生成新的sRT矩阵: 构建一个新的4x4的变换矩阵aligned_sRT，其中旋转部分乘以缩放因子s，平移部分保持不变。这个新矩阵就是对齐后的变换矩阵。
返回值
    aligned_sRT: 经过旋转对齐处理后的新变换矩阵，依然保持4x4的结构，但旋转部分已按照对称性进行了规范化处理，便于后续的分析或比较。
总的来说，该函数主要用于解决具有旋转对称性的3D对象姿态估计中旋转矩阵表达不唯一的问题，通过特定的旋转对齐操作，使模型姿态表示更加一致和合理。
    """
    s = np.cbrt(np.linalg.det(sRT[:3, :3]))
    R = sRT[:3, :3] / s
    T = sRT[:3, 3]

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x ** 2 + theta_y ** 2)
    s_map = np.array([[theta_x / r_norm, 0.0, -theta_y / r_norm],
                      [0.0, 1.0, 0.0],
                      [theta_y / r_norm, 0.0, theta_x / r_norm]])
    rotation = R @ s_map
    aligned_sRT = np.identity(4, dtype=np.float32)
    aligned_sRT[:3, :3] = s * rotation
    aligned_sRT[:3, 3] = T
    return aligned_sRT


def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    # draw pillars in minor darker color
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

    return img


def draw_detections(img, out_dir, data_name, img_id, intrinsics, pred_sRT, pred_size, pred_class_ids,
                    gt_sRT, gt_size, gt_class_ids, nocs_sRT, nocs_size, nocs_class_ids, draw_gt=True, draw_nocs=True):
    """ Visualize pose predictions.

    这段代码 draw_detections 是计算机视觉任务中用于可视化物体姿态预测的一部分。
    此函数在原始图像上叠加检测到的物体边界框及其预测姿态、真实姿态（ground truth），以及如果可用的话，NOCS（标准化物体坐标空间）姿态。
    以下是各参数的简要说明：
    img：要在其上绘制检测结果的原始图像。
    out_dir：保存输出图像的目录。
    data_name：数据集名称的字符串。
    img_id：该图像在数据集中的标识符。
    intrinsics：相机内参矩阵，用于将3D点投影到2D平面上。
    pred_sRT、gt_sRT、nocs_sRT：这些分别代表被检测物体的预测姿态、真实姿态和NOCS姿态（尺度和旋转矩阵）。
    pred_size、gt_size、nocs_size：这些姿态下物体的尺寸。
    pred_class_ids、gt_class_ids、nocs_class_ids：对应物体的类别ID。
    draw_gt 和 draw_nocs：布尔标志，指示是否绘制真实姿态和NOCS姿态。
    在函数中，它首先定义了输出图像的路径。然后，根据参数 draw_nocs 和 draw_gt 的值，它可能会用蓝色绘制NOCS结果，用绿色绘制真实姿态。
    最后，它总是用红色绘制预测姿态。每个姿态都是通过计算3D边界框，应用相应的旋转和平移变换，然后将其投影到2D平面上来实现的。
    这通常用于评估姿态估计算法的性能，使预测、真实值和NOCS值之间的差异一目了然。
    """
    out_path = os.path.join(out_dir, '{}_{}_pred.png'.format(data_name, img_id))

    # draw nocs results - BLUE color
    if draw_nocs:
        for i in range(nocs_sRT.shape[0]):
            if nocs_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(nocs_sRT[i, :, :])
            else:
                sRT = nocs_sRT[i, :, :]
            bbox_3d = get_3d_bbox(nocs_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (255, 0, 0))
    # darw ground truth - GREEN color
    if draw_gt:
        for i in range(gt_sRT.shape[0]):
            if gt_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(gt_sRT[i, :, :])
            else:
                sRT = gt_sRT[i, :, :]
            bbox_3d = get_3d_bbox(gt_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (0, 255, 0))
    # darw prediction - RED color
    for i in range(pred_sRT.shape[0]):
        if pred_class_ids[i] in [1, 2, 4]:
            sRT = align_rotation(pred_sRT[i, :, :])
        else:
            sRT = pred_sRT[i, :, :]
        bbox_3d = get_3d_bbox(pred_size[i, :], 0)
        transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        img = draw_bboxes(img, projected_bbox, (0, 0, 255))

    cv2.imwrite(out_path, img)
    # cv2.imshow('vis', img)
    # cv2.waitKey(0)