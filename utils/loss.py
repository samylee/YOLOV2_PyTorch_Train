import torch
import torch.nn as nn
import numpy as np
from utils.utils import bbox_iou


class YOLOV2Loss(nn.Module):
    def __init__(self, B=5, C=20, object_scale=5, noobject_scale=1, class_scale=1, coord_scale=1, anchors=None, device=None):
        super(YOLOV2Loss, self).__init__()
        self.B, self.C = B, C
        self.object_scale, self.noobject_scale, self.class_scale, self.coord_scale = object_scale, noobject_scale, class_scale, coord_scale
        self.prior_coord_scale = 0.01
        self.device = device
        self.anchors = torch.from_numpy(np.asarray(anchors, dtype=np.float32)).view(-1, 2).to(self.device)
        self.iou_thresh = 0.6
        self.seen = 0

        self.class_criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.noobj_criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.obj_criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.coords_criterion = nn.MSELoss(reduction='sum').to(self.device)
        self.prior_coords_criterion = nn.MSELoss(reduction='sum').to(self.device)

    def target_box_encode(self, box, grid_size, i, j, index):
        box_out = box.clone()
        box_out[0] = box[0] * grid_size - i
        box_out[1] = box[1] * grid_size - j
        box_out[2] = torch.log(box[2] * grid_size / self.anchors[index, 0])
        box_out[3] = torch.log(box[3] * grid_size / self.anchors[index, 1])
        return box_out

    def forward(self, preds, targets):
        batch_size, _, grid_size, _ = preds.shape

        # num_samples, 13(grid), 13(grid), 5(anchors), 25 (tx, ty, tw, th, conf, classes)
        output_permute = (
            preds.permute(0, 2, 3, 1)
                .view(batch_size, grid_size, grid_size, self.B, self.C + 5)
                .contiguous()
        )

        # tx, ty, tw, th
        preds_tx = torch.sigmoid(output_permute[..., 0])
        preds_ty = torch.sigmoid(output_permute[..., 1])
        preds_tw = output_permute[..., 2]
        preds_th = output_permute[..., 3]
        preds_xywh = torch.cat(
            (preds_tx.unsqueeze(-1),
             preds_ty.unsqueeze(-1),
             preds_tw.unsqueeze(-1),
             preds_th.unsqueeze(-1)),
            -1
        )
        # conf
        preds_conf = torch.sigmoid(output_permute[..., 4])
        # class
        preds_class = torch.softmax(output_permute[..., 5:], dim=-1)

        # decode boxes for forward
        preds_coords = torch.empty((batch_size, grid_size, grid_size, self.B, 4), dtype=torch.float32).to(self.device)
        preds_coords[..., 0] = (preds_tx + torch.arange(grid_size).repeat(grid_size, 1).view([1, grid_size, grid_size, 1]).float().to(self.device)) / grid_size
        preds_coords[..., 1] = (preds_ty + torch.arange(grid_size).repeat(grid_size, 1).t().view([1, grid_size, grid_size, 1]).float().to(self.device)) / grid_size
        preds_coords[..., 2] = (torch.exp(preds_tw) * self.anchors[:, 0].view(1, 1, 1, self.B)) / grid_size
        preds_coords[..., 3] = (torch.exp(preds_th) * self.anchors[:, 1].view(1, 1, 1, self.B)) / grid_size

        # find noobj
        noobj_mask_list = []
        for b in range(batch_size):
            preds_coords_batch = preds_coords[b]
            targets_coords_batch = targets[b, :, 1:]
            preds_ious_list = []
            for target_coords_batch in targets_coords_batch:
                if target_coords_batch[0] == 0:
                    break
                preds_ious_noobj = bbox_iou(preds_coords_batch.view(-1, 4), target_coords_batch.unsqueeze(0)).view(grid_size, grid_size, self.B)
                preds_ious_list.append(preds_ious_noobj)
            preds_ious_tensor = torch.stack(preds_ious_list, dim=-1)
            preds_ious_max = torch.max(preds_ious_tensor, dim=-1)[0]

            noobj_mask = preds_ious_max <= self.iou_thresh
            noobj_mask_list.append(noobj_mask)
        noobj_mask_tensor = torch.stack(noobj_mask_list)

        # generate obj mask and iou
        obj_mask_tensor = torch.empty_like(preds_conf, dtype=torch.bool, requires_grad=False).fill_(False).to(self.device)
        obj_ious_tensor = torch.zeros_like(preds_conf, requires_grad=False)
        obj_class_tensor = torch.zeros_like(preds_class, requires_grad=False)
        targets_coords_encode = torch.zeros_like(preds_coords, requires_grad=False)
        loss_coords_wh_scale = torch.zeros_like(preds_coords, requires_grad=False)
        for b in range(batch_size):
            preds_coords_batch = preds_coords[b]
            targets_batch = targets[b]
            for target_batch in targets_batch:
                target_class_batch = int(target_batch[0])
                assert target_class_batch < self.C, 'oh shit'
                target_coords_batch = target_batch[1:]
                if target_coords_batch[0] == 0:
                    break
                i = int(target_coords_batch[0] * grid_size)
                j = int(target_coords_batch[1] * grid_size)

                target_coords_batch_shift = torch.zeros((1, 4), dtype=torch.float32).to(self.device)
                target_coords_batch_shift[0, 2:] = target_coords_batch[2:]

                anchors_match_batch_shift = torch.zeros((self.B, 4), dtype=torch.float32).to(self.device)
                anchors_match_batch_shift[:, 2] = self.anchors[:, 0] / grid_size
                anchors_match_batch_shift[:, 3] = self.anchors[:, 1] / grid_size

                anchors_ious = bbox_iou(anchors_match_batch_shift, target_coords_batch_shift)

                # get obj index
                anchors_ious_index = torch.max(anchors_ious, dim=0)[1].item()
                # get obj real iou
                preds_ious_real = bbox_iou(preds_coords_batch[j, i, anchors_ious_index], target_coords_batch).item()
                # target box encode
                target_coords_batch_encode = self.target_box_encode(target_coords_batch, grid_size, i, j, anchors_ious_index)

                # ignore second label in the same grid and same anchor_index
                if obj_mask_tensor[b, j, i, anchors_ious_index]:
                    continue

                obj_mask_tensor[b, j, i, anchors_ious_index] = True
                noobj_mask_tensor[b, j, i, anchors_ious_index] = False
                obj_class_tensor[b, j, i, anchors_ious_index, target_class_batch] = 1.0
                obj_ious_tensor[b, j, i, anchors_ious_index] = preds_ious_real
                targets_coords_encode[b, j, i, anchors_ious_index] = target_coords_batch_encode

                current_scale_wh = 2 - target_coords_batch[2]*target_coords_batch[3]
                loss_coords_wh_scale[b, j, i, anchors_ious_index] = current_scale_wh.repeat(4)

        # 1. noobj loss
        preds_conf_noobj_mask = preds_conf[noobj_mask_tensor]
        loss_noobj = self.noobj_criterion(preds_conf_noobj_mask, torch.zeros_like(preds_conf_noobj_mask))

        # 2. obj loss
        preds_conf_obj_mask = preds_conf[obj_mask_tensor]
        preds_iou_obj_mask = obj_ious_tensor[obj_mask_tensor]
        loss_obj = self.obj_criterion(preds_conf_obj_mask, preds_iou_obj_mask)

        # 3. class loss
        class_mask = obj_mask_tensor.unsqueeze(-1).expand_as(preds_class)
        preds_class_mask = preds_class[class_mask]
        obj_class_tensor_mask = obj_class_tensor[class_mask]
        loss_class = self.class_criterion(preds_class_mask, obj_class_tensor_mask)

        # 4. coords loss
        coords_mask = obj_mask_tensor.unsqueeze(-1).expand_as(preds_coords)
        preds_coords_obj_mask = preds_xywh[coords_mask]
        targets_coords_obj_encode_mask = targets_coords_encode[coords_mask]
        loss_coords_wh_scale_mask = loss_coords_wh_scale[coords_mask]
        loss_coords = self.coords_criterion(preds_coords_obj_mask * loss_coords_wh_scale_mask, targets_coords_obj_encode_mask * loss_coords_wh_scale_mask)

        # anchors targets for seen<12800
        if self.seen < 12800:
            prior_coords = torch.zeros_like(preds_coords, requires_grad=False)
            prior_coords[..., 0] = 0.5
            prior_coords[..., 1] = 0.5
            preds_coords_noobj_mask = preds_xywh[~coords_mask]
            targets_coords_noobj_encode_mask = prior_coords[~coords_mask]
            loss_prior_coords = self.prior_coords_criterion(preds_coords_noobj_mask, targets_coords_noobj_encode_mask)
        else:
            loss_prior_coords = 0

        loss = self.class_scale * loss_class + \
               self.object_scale * loss_obj + \
               self.noobject_scale * loss_noobj + \
               self.coord_scale * loss_coords + \
               self.prior_coord_scale * loss_prior_coords

        self.seen += batch_size
        return loss/batch_size