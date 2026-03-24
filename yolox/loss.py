#!/usr/bin/env python
"""
Better YOLOX loss with proper target assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils.ppe_dataset import PPE_DATA


class YOLOXLoss(nn.Module):
    """YOLOX Loss with SimOTA assignment (simplified)"""
    
    def __init__(self, num_classes, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # TODO should I use this? Where would it go?
        # also TODO should I use focal loss instead of BCE?
        self.l1_loss = nn.L1Loss(reduction='none')
    
    # TODO test this 
    def forward(self, predictions, targets, input_size=640):
        device = predictions.device
        batch_size = predictions.shape[0]

        x_shifts, y_shifts, expanded_strides, centers = self.build_anchor_meta(
            input_size=input_size, device=device
        )

        total_loss = 0
        total_obj_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        total_num_fg = 0

        for batch_idx in range(batch_size):
            valid_mask = targets[batch_idx, :, 0] >= 0
            if not valid_mask.any():
                obj_targets = torch.zeros(predictions.shape[1], device=device)
                obj_loss = self.bce_loss(predictions[batch_idx, :, 4], obj_targets).mean()
                total_obj_loss += obj_loss
                continue

            gt_targets = targets[batch_idx, valid_mask]
            gt_classes = gt_targets[:, 0].long()
            gt_boxes = gt_targets[:, 1:5] * input_size

            num_gt = len(gt_boxes)

            (gt_matched_classes,
             pos_mask,
             pred_ious_this_matching,
             matched_gt_inds,
             batch_num_fg) = self.get_assignments(
                batch_idx,
                num_gt,
                gt_boxes,
                gt_classes,
                predictions[batch_idx, :, :4],
                expanded_strides,
                x_shifts,
                y_shifts,
                predictions[:, :, 5:],
                predictions[:, :, 4:5],
            )

            total_num_fg += batch_num_fg
            num_pos = pos_mask.sum()

            obj_targets = torch.zeros(predictions.shape[1], device=device)
            obj_targets[pos_mask] = 1.0
            obj_loss = self.bce_loss(predictions[batch_idx, :, 4], obj_targets).mean()
            total_obj_loss += obj_loss

            if num_pos > 0:
                cls_targets = torch.zeros(num_pos, self.num_classes, device=device)
                cls_targets[range(num_pos), gt_classes[matched_gt_inds]] = 1.0
                cls_loss = self.bce_loss(
                    predictions[batch_idx, pos_mask, 5:], cls_targets
                ).mean()
                total_cls_loss += cls_loss

                pred_boxes = predictions[batch_idx, pos_mask, :4]
                target_boxes = gt_boxes[matched_gt_inds]
                box_loss = self.ciou_loss(pred_boxes, target_boxes).mean()
                total_box_loss += box_loss

        total_loss = (5.0 * total_box_loss + total_obj_loss + total_cls_loss) / max(total_num_fg, 1)
        return {
            'total_loss': total_loss,
            'box_loss': total_box_loss,
            'obj_loss': total_obj_loss,
            'cls_loss': total_cls_loss,
            'num_fg': total_num_fg
        }

    def build_anchor_meta(self, input_size: int, device: torch.device):
        x_list, y_list, s_list, c_list = [], [], [], []

        for stride in self.strides:
            feat_size = input_size // stride
            yv, xv = torch.meshgrid(
                torch.arange(feat_size, device=device),
                torch.arange(feat_size, device=device),
                indexing="ij",
            )
            xv = xv.flatten()
            yv = yv.flatten()

            x_list.append(xv)
            y_list.append(yv)
            s_list.append(torch.full_like(xv, stride).float())
            c_list.append(torch.stack((xv.float() + 0.5, yv.float() + 0.5), dim=1) * stride)

        x_shifts = torch.cat(x_list).unsqueeze(0)
        y_shifts = torch.cat(y_list).unsqueeze(0)
        expanded_strides = torch.cat(s_list).unsqueeze(0)
        centers = torch.cat(c_list, dim=0)

        return x_shifts, y_shifts, expanded_strides, centers
   
    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
    ):

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )


        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # this is for mixed precision training, could be taken off if I leave everything
        # in fp32
        # TODO figure out if I am running in mixed precision or not
        with torch.amp.autocast(device_type="cuda",enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
        return area_i / (area_a[:, None] + area_b - area_i)

    def ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps=1e-7):
        """
        pred_boxes & target_boxes:  (N, 4)  format  [cx, cy, w, h]  (absolute pixels)
        Returns:  CIoU loss  (N,)  where 0 == perfect overlap
        """
        # p is for predicted, g is for ground truth. Extract their
        # centers, widths, and heights
        px, py, pw, ph = pred_boxes.unbind(-1)
        gx, gy, gw, gh = target_boxes.unbind(-1)

        # Convert to corners
        pred_x1, pred_y1 = px - pw / 2, py - ph / 2
        pred_x2, pred_y2 = px + pw / 2, py + ph / 2
        gt_x1,  gt_y1  = gx - gw / 2, gy - gh / 2
        gt_x2,  gt_y2  = gx + gw / 2, gy + gh / 2

        # compute their intersection
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                    (inter_y2 - inter_y1).clamp(min=0)

        # Areas
        pred_area = pw * ph
        gt_area   = gw * gh
        union     = pred_area + gt_area - inter_area 

        iou = inter_area / (union + eps)                              # term 1

        # ------ DIoU term ------
        #TODO understand and review the formulas for DIoU and CIoU
        center_dist_sq = (px - gx)**2 + (py - gy)**2          # ρ²
        enclose_x1 = torch.min(pred_x1, gt_x1)
        enclose_y1 = torch.min(pred_y1, gt_y1)
        enclose_x2 = torch.max(pred_x2, gt_x2)
        enclose_y2 = torch.max(pred_y2, gt_y2)
        c2 = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 
        diou_term = center_dist_sq / (c2 + eps)                      # term 2

        # ------ aspect-ratio term ------
        v = (4 / (torch.pi ** 2)) * torch.pow(
                torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps)), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)                   # eq-10 in paper
        ciou = iou - diou_term - alpha * v                    # IoU *minus* penalties
        loss = 1 - ciou                                       # so 0 is best
        return loss
