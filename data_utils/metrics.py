import torch
from yolox.model import create_yolox_s
from yolox.handle_weights import load_pretrained_weights

# pred: (6,) gt: (num_gt, 5). For both, 1:5 are box coordinates
def pairwise_iou(pred, gt):
    '''Calculate the Intersection over Union (IoU) between a predicted box and ground truth boxes.

    :param pred: The predicted bounding box, shape (6,). Indices 1:5 are box coordinates.
    :type pred: torch.Tensor
    :param gt: Ground truth bounding boxes, shape (num_gt, 5). Indices 1:5 are box coordinates.
    :type gt: torch.Tensor
    :return: IoU values between the predicted box and each ground truth box.
    :rtype: torch.Tensor
    '''

    # Extract coordinates, these are expected in x1, y1, x2, y2 format
    px1, py1, px2, py2 = pred[1:5]

    gcx = gt[:, 1]
    gcy = gt[:, 2]
    gw = gt[:, 3]
    gh = gt[:, 4]
    gx1, gy1 = gcx - gw / 2, gcy - gh / 2
    gx2, gy2 = gcx + gw / 2, gcy + gh / 2
    gx1 *= 640; gx2 *= 640
    gy1 *= 640; gy2 *= 640

    # calculate intersection
    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # calculate union
    pw = px2 - px1
    ph = py2 - py1
    pred_area = pw * ph
    gt_area = gw * gh
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + 1e-7)
    return iou

def calculate_AP_per_class(gt: torch.Tensor, preds: torch.Tensor, gt_to_img: torch.Tensor, 
                           preds_to_img: torch.Tensor, iou_thresh: float, 
                           device: str = "cuda"):
    '''Calculate Average Precision (AP), precision, and recall for a single class.

    :param gt: Ground truth boxes for the class.
    :type gt: torch.Tensor
    :param preds: Predicted boxes for the class.
    :type preds: torch.Tensor
    :param gt_to_img: Mapping from ground truth boxes to image IDs.
    :type gt_to_img: torch.Tensor
    :param preds_to_img: Mapping from predicted boxes to image IDs.
    :type preds_to_img: torch.Tensor
    :param iou_thresh: IoU threshold for matching predictions to ground truth.
    :type iou_thresh: float
    :param device: Device to run calculations on. Defaults to "cuda".
    :type device: str
    :return: AP, precision, and recall for the class.
    :rtype: tuple
    '''

    # initialize the true positive, false positive, precision and recall for accumulating metrics
    tp = torch.zeros(preds.shape[0], dtype=torch.float32).to(device)
    fp = torch.zeros(preds.shape[0], dtype=torch.float32).to(device)
    precision = torch.zeros(preds.shape[0], dtype=torch.float32).to(device)
    recall = torch.zeros(preds.shape[0], dtype=torch.float32).to(device)

    num_gt = gt.shape[0]
    gt_matched = torch.zeros(gt.shape[0], dtype = torch.bool).to(device)
    for i, pred in enumerate(preds):
        # index into preds_to img to find which image this prediction is a part of,
        # then extract all ground truth boxes from this image to compute IoU
        pred_img = preds_to_img[i]
        img_mask = (gt_to_img == pred_img) & (~gt_matched)
        same_img_gt = gt[img_mask]
        if same_img_gt.shape[0] == 0:
            fp[i] = 1
            continue

        pairwise_ious = pairwise_iou(pred, same_img_gt)
        # find the best IoU match
        best_iou, best_gt_idx = torch.max(pairwise_ious, dim=0)
        if best_iou >= iou_thresh:
            # mark this ground truth box as used
            gt_matched_idx = torch.nonzero(img_mask)[best_gt_idx] 
            gt_matched[gt_matched_idx] = True
            tp[i] = 1
        else:
            fp[i] = 1
    # calculate precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
    recall = tp_cumsum / (float(num_gt) + 1e-9)

    # how precision recall curve should be calculated, according to chat
    mrec = torch.cat([torch.tensor([0.], device=recall.device),
                  recall,
                  torch.tensor([1.], device=recall.device)])

    mpre = torch.cat([torch.tensor([0.], device=precision.device),
                  precision,
                  torch.tensor([0.], device=precision.device)])
    # reverse, cumulative max, then flip back – vectorised, no Python loop
    mpre = torch.flip(torch.cummax(torch.flip(mpre, dims=[0]), dim=0).values,
                  dims=[0])
    chg = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze(1)
    if chg.numel() == 0:
        # no change in recall, return 0 AP
        return torch.tensor(0.0, device=device)
    ap  = torch.sum((mrec[chg + 1] - mrec[chg]) * mpre[chg + 1])
    TP = tp.sum().item()
    FP = fp.sum().item()
    precision_item = TP / (TP + FP + 1e-9)
    recall_item = TP / (num_gt + 1e-9)
    return ap, precision_item, recall_item

# gt is a list of the ground truth boxes, preds is a list of predicted boxes+confidence
def calculate_mAP(img_ids: torch.Tensor, gts: torch.Tensor, preds: torch.Tensor, 
                  num_classes = 4, iou_thresh = 0.5, 
                  device = "cuda", writer = None, epoch = 0):
    '''Calculate mean Average Precision (mAP) across all classes.

    :param img_ids: Tensor of image IDs.
    :type img_ids: torch.Tensor
    :param gts: Ground truth boxes for all images.
    :type gts: torch.Tensor
    :param preds: Predicted boxes for all images.
    :type preds: torch.Tensor
    :param num_classes: Number of classes. Defaults to 4.
    :type num_classes: int
    :param iou_thresh: IoU threshold for matching predictions to ground truth. Defaults to 0.5.
    :type iou_thresh: float
    :param device: Device to run calculations on. Defaults to "cuda".
    :type device: str
    :param writer: Optional writer for logging metrics. Defaults to None.
    :type writer: object
    :param epoch: Epoch number for logging. Defaults to 0.
    :type epoch: int
    :return: Mean Average Precision (mAP) across all classes.
    :rtype: float
    '''

    # a mapping from ground truth boxes to their img id
    # used to ensure predictions are being compared only to the same image

    # shape (batch, max_gt)
    gt_mask = gts[:, :, 0] >= 0
    gt_to_img = torch.repeat_interleave(img_ids, gt_mask.sum(1), dim=0)
    true_gt = gts[gt_mask]
    # ngt = num ground truth, npk = num predictions kept

    # refactor this to work with the way batch_nms returns indices 
    batch_len = img_ids.shape[0]
    preds_to_img = torch.empty(0).to(device)
    final_preds = torch.empty(0, 6).to(device)  # 6 = 1 class + 4 bbox coords + 1 score
    for batch_idx in range(batch_len):
        # add img id to the beginning of each ground truth box
        img_id = img_ids[batch_idx]
        pred = preds[batch_idx]
        processed_preds = post_process_img(pred, confidence_threshold=0.25, iou_threshold=iou_thresh)
        preds_to_img = torch.cat((preds_to_img, img_id.repeat(processed_preds.shape[0])), dim=0)
        final_preds = torch.cat((final_preds, processed_preds), dim=0)
    scores= final_preds[:, -1]
    # sort by score, descending
    _, order = scores.sort(descending=True)
    final_preds = final_preds[order]
    preds_to_img = preds_to_img[order]

    total_ap = 0
    classes = ["Ascomycetes/Ascospores", "Basidiomycetes/Basidiospores", "Cladosporium", "Hyphal fragments", "Non-specified Spore",
    "Penicillium/Aspergillus-like", "Rusts/Smuts/Periconia/Myxomycetes", "Debris",]
    total_precision = 0.0
    total_recall = 0.0
    for i in range(num_classes):
        idx = torch.tensor([i], device=device)
        gt_class_mask = true_gt[:, 0] == idx
        pred_class_mask = final_preds[:, 0] == idx
        if not gt_class_mask.any() and not pred_class_mask.any():
            continue
        AP, precision, recall = calculate_AP_per_class(true_gt[gt_class_mask], final_preds[pred_class_mask], 
                               gt_to_img[gt_class_mask], preds_to_img[pred_class_mask],
                               iou_thresh, device=device)
        if writer is not None:
            writer.add_scalar(f"AP/{classes[i]}", AP, global_step=epoch)
            writer.add_scalar(f"Precision/{classes[i]}", precision, global_step=epoch)
            writer.add_scalar(f"Recall/{classes[i]}", recall, global_step=epoch)
        total_ap += AP
        total_precision += precision
        total_recall += recall

    if writer is not None:
        writer.add_scalar("Precision/Overall", total_precision / num_classes, global_step=epoch)
        writer.add_scalar("Recall/Overall", total_recall / num_classes, global_step=epoch)
    return total_ap / num_classes

def post_process_img(output: torch.Tensor, confidence_threshold: float = 0.25, iou_threshold: float = 0.5) -> torch.Tensor:
    '''This function expects the output to be in pixel values and sigmoid to already be applied
    to obj and class probabilities.

    :param torch.Tensor output: The model's output on a given image.
    :param float confidence_threshold: The confidence threshold for filtering predictions. Defaults to 0.25
    :param float iou_threshold: The IoU threshold for filtering predictions. Defaults to 0.5
    :return torch.Tensor: The processed predictions in x1 y1 x2 y2 format (top left and bottom right points)
    '''
    x1 = output[..., 0:1] - output[..., 2:3] / 2
    y1 = output[..., 1:2] - output[..., 3:4] / 2
    x2 = output[..., 0:1] + output[..., 2:3] / 2
    y2 = output[..., 1:2] + output[..., 3:4] / 2

    # boxes: (batch, num_anchors, 4)
    boxes = torch.cat([x1, y1, x2, y2], dim=-1)

    # (batch, num_anchors, 1)
    obj = output[..., 4:5]
    class_probs = output[..., 5:]

    scores = obj * class_probs
    best_scores, best_class = scores.max(dim=-1)

    mask = best_scores > confidence_threshold
    best_scores = best_scores[mask] 
    best_class = best_class[mask] 
    boxes = boxes[mask]
    keep = nms(boxes, best_scores, iou_threshold = iou_threshold)
    final_boxes = boxes[keep]
    final_classes = best_class[keep]
    final_scores = best_scores[keep]
    # final classes and final scores have shape (num_kept,), so unsqueeze to add the dim 1 again
    predictions = torch.cat((final_classes.unsqueeze(1), 
                             final_boxes, 
                             final_scores.unsqueeze(1)), dim=1)
    return predictions


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    '''Compute the IoU matrix between two sets of axis-aligned bounding boxes.

    :param boxes1: First set of boxes in (x1, y1, x2, y2) format.
    :type boxes1: torch.Tensor
    :param boxes2: Second set of boxes in (x1, y1, x2, y2) format.
    :type boxes2: torch.Tensor
    :return: IoU matrix of shape (N, M) where N and M are the number of boxes in boxes1 and boxes2.
    :rtype: torch.Tensor
    '''
    """
    Vectorized IoU for two -sets- of axis-aligned boxes.
    boxes{1,2}: (N, 4) or (M, 4) in XYXY format (x1, y1, x2, y2)
    Returns:    (N, M) IoU matrix
    """
    # areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(0)

    # pairwise intersections
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)                             # width‑height
    inter = wh[..., 0] * wh[..., 1]                         # (N, M)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    '''Perform Non-Maximum Suppression (NMS) on bounding boxes.

    :param boxes: Bounding boxes in (x1, y1, x2, y2) format.
    :type boxes: torch.Tensor
    :param scores: Confidence scores for each box.
    :type scores: torch.Tensor
    :param iou_threshold: IoU threshold for suppressing overlapping boxes.
    :type iou_threshold: float
    :return: Indices of boxes that survive NMS, sorted by descending score.
    :rtype: torch.Tensor
    '''
    """
    Pure-PyTorch Non-Maximum Suppression mirroring
    torchvision.ops.nms(...).

    Args
    ----
    boxes         (Tensor[N,4])  - boxes in (x1, y1, x2, y2) format
    scores        (Tensor[N])     - confidence scores
    iou_threshold (float)         - IoU overlap threshold to suppress

    Returns
    -------
    keep (Tensor[K]) - indices of boxes that survive NMS,
                       sorted in descending score order
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # sort by score descending
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]              # index of current highest score
        keep.append(i.item())

        if order.numel() == 1:    # nothing left to compare
            break

        # IoU of the current box with the rest
        ious = _box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)

        # keep boxes with IoU ≤ threshold
        order = order[1:][ious <= iou_threshold]

    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)

