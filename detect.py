import torch
import cv2

from models.YOLOV2 import YOLOV2
from utils.utils import nms


def postprocess(output, num_classes, anchors, conf_thresh, img_w, img_h):
    num_samples, _, grid_size, _ = output.shape
    num_anchors = anchors.size(0)

    # num_samples, 5(anchors), 13(grid), 13(grid), 25 (tx, ty, tw, th, conf, classes)
    output_permute = (
        output.permute(0, 2, 3, 1)
            .view(num_samples, grid_size, grid_size, num_anchors, num_classes + 5)
            .contiguous()
    )

    # tx, ty, tw, th
    pred_tx = torch.sigmoid(output_permute[..., 0])
    pred_ty = torch.sigmoid(output_permute[..., 1])
    pred_tw = output_permute[..., 2]
    pred_th = output_permute[..., 3]
    # conf
    pred_conf = torch.sigmoid(output_permute[..., 4])
    # classes
    pred_cls = torch.softmax(output_permute[..., 5:], dim=-1)

    # decode boxes
    pred_coords = torch.empty((num_samples, grid_size, grid_size, num_anchors, 4), dtype=torch.float32)
    pred_coords[..., 0] = pred_tx + torch.arange(grid_size).repeat(grid_size, 1).view([1, grid_size, grid_size, 1]).float()
    pred_coords[..., 1] = pred_ty + torch.arange(grid_size).repeat(grid_size, 1).t().view([1, grid_size, grid_size, 1]).float()
    pred_coords[..., 2] = torch.exp(pred_tw) * anchors[:, 0].view(1, 1, 1, num_anchors)
    pred_coords[..., 3] = torch.exp(pred_th) * anchors[:, 1].view(1, 1, 1, num_anchors)

    # num_samples, (s*s*a), (C+5)
    predictions = torch.cat(
        (
            pred_coords.view(num_samples, -1, 4) / grid_size,
            pred_conf.view(num_samples, -1, 1),
            pred_cls.view(num_samples, -1, num_classes),
        ),
        -1,
    )

    # for num_samples = 1, (s*s*a), (C+5)
    predictions = predictions.squeeze(0)

    # Filter out confidence scores below conf_thresh
    detections = predictions[predictions[:, 4] >= conf_thresh].clone()
    if not detections.size(0):
        return detections

    # conf * classes
    class_confs, class_id = detections[:, 5:].max(1, keepdim=True)
    class_confs *= detections[:, 4].unsqueeze(-1)

    # xywh to xyxy
    detections_cp = detections[:, :4].clone()
    detections[:, 0] = (detections_cp[:, 0] - detections_cp[:, 2] / 2.) * img_w
    detections[:, 1] = (detections_cp[:, 1] - detections_cp[:, 3] / 2.) * img_h
    detections[:, 2] = (detections_cp[:, 0] + detections_cp[:, 2] / 2.) * img_w
    detections[:, 3] = (detections_cp[:, 1] + detections_cp[:, 3] / 2.) * img_h

    return torch.cat((detections[:, :4], class_confs.float(), class_id.float()), 1)


def preprocess(img, net_w, net_h):
    # img bgr2rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize img
    img_resize = cv2.resize(img_rgb, (net_w, net_h))

    # norm img
    img_resize = torch.from_numpy(img_resize.transpose((2, 0, 1)))
    img_norm = img_resize.float().div(255).unsqueeze(0)
    return img_norm


def model_init(model_path, B=5, C=20):
    # load moel
    model = YOLOV2(B=B, C=C)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


if __name__ == '__main__':
    # load moel
    checkpoint_path = 'weights/yolov2_final.pth'
    B, C = 5, 20
    model = model_init(checkpoint_path, B, C)

    # params init
    net_w, net_h = 416, 416
    anchors = torch.tensor([[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]],
                           dtype=torch.float32)
    thresh = 0.5
    iou_thresh = 0.45
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    # load img
    img = cv2.imread('demo/000004.jpg')
    img_h, img_w, _ = img.shape

    # preprocess
    img_norm = preprocess(img, net_w, net_h)

    # forward
    output = model(img_norm)

    # postprocess
    results = postprocess(output, C, anchors, thresh, img_w, img_h)

    if results.size(0) > 0:
        # nms
        results = nms(results.data.cpu().numpy(), iou_thresh)

        # show
        for i in range(results.shape[0]):
            cv2.rectangle(img, (int(results[i][0]), int(results[i][1])), (int(results[i][2]), int(results[i][3])), (0,255,0), 2)
            cv2.putText(img, classes[int(results[i][5])] + '-' + str(round(results[i][4], 4)), (int(results[i][0]), int(results[i][1])), 0, 0.6, (0,255,255), 2)

    # cv2.imwrite('assets/result4.jpg', img)
    cv2.imshow('demo', img)
    cv2.waitKey(0)
