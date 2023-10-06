# -*- coding:utf-8 -*-
"""
Author: LJC
Email: jiacheng_li@std.uestc.edu.cn
name: dec_fusion_multispectral.py
Data: 2021.07.26
"""
import copy

import torch
import cv2
import numpy as np

import os
from utils import SSIM
from utils import fda
from PIL import Image


def letterbox(img, new_shape=(64, 64), color=(128, 128, 128), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def intersect(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def contain(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]

    return inter / area_a  # inter/A

def Dec_Fusion_Multispectral(pred_ref,pred_sen,iou_thres,contain_thres):
    if len(pred_ref[0]) == 0:
        pred_fuse = copy.deepcopy(pred_sen)
    elif len(pred_sen[0]) == 0:
        pred_fuse = copy.deepcopy(pred_ref)
    else:
        pred_fuse = copy.deepcopy(pred_ref)
        for i in range(len(pred_fuse[0])):
            pred_fuse[0][i][5] = 1

        fuse_iou = jaccard(pred_sen[0][:, :4], pred_ref[0][:, :4])
        contain_iou_1 = contain(pred_sen[0][:, :4], pred_ref[0][:, :4])
        contain_iou_2 = contain(pred_ref[0][:, :4], pred_sen[0][:, :4])
        contain_iou = torch.max(contain_iou_1, contain_iou_2.t())
        m, n = fuse_iou.shape
        for i in range(m):
            add = True
            temp4 = 0
            for j in range(n):
                if fuse_iou[i][j] > iou_thres or contain_iou[i][j] > contain_thres:
                    print('Fuse_Done_Better')
                    add = False
                    if (pred_sen[0][i][4] > pred_ref[0][j][4]) and (fuse_iou[i][j] > temp4):
                        temp4 = fuse_iou[i][j]
                        pred_fuse[0][j][4:6] = pred_sen[0][i][4:6]
                        pred_fuse[0][j][5] = 0
                    else:
                        pred_fuse[0][j][5] = 0

            if add is True:
                print('Fuse_Done_Add')
                temp5 = copy.deepcopy(pred_sen[0][i])
                temp5[5] = 0
                pred_fuse[0] = torch.cat([pred_fuse[0], torch.unsqueeze(temp5, 0)], dim=0)
    return pred_fuse


def box2windows(opt, x1, x2, y1, y2, window_h_max, window_w_max, stride):

    sample_name = os.path.split(opt.main_dir)[-1]
    larger_range_sample_list =['07']
    if sample_name in larger_range_sample_list:
        num = 20
    else:
        num = 15        #15
    window_h_start = int(max(y1 - num * stride, 0))
    window_h_end = int(min(y2 + num * stride, window_h_max))
    window_w_start = int(max(x1 - num * stride, 0))
    window_w_end = int(min(x2 + num * stride, window_w_max))
    oriw_loc = num
    orih_loc = num

    return window_h_start, window_h_end, window_w_start, window_w_end, oriw_loc, orih_loc

def ssim_struct_caculation(x,y):
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    y = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    ret = SSIM.ssim_struct(x, y)
    return ret

def img_prepro(opt, reference_img, sensed_img, img_name):       #sensed:ir
    ref_img = copy.deepcopy(reference_img)
    sen_img = 255 - copy.deepcopy(sensed_img)
    ref_img = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB))
    sen_img = Image.fromarray(cv2.cvtColor(sen_img, cv2.COLOR_GRAY2RGB))

    reference_pro = fda.FDA_source_to_target_np(ref_img, sen_img, L=0.01)

    sensed_pro = 255 - sensed_img
    reference_pro = reference_pro[:, :, 0]

    for i in range(reference_pro.shape[0]):
        for j in range(reference_pro.shape[1]):
            if reference_pro[i][j] > 255:
                reference_pro[i][j] = 255

    reference_pro = reference_pro.astype(np.uint8)

    return reference_pro, sensed_pro

def calculate_similarity(opt, x, y):

    x = (x - x.min())/(x.max() - x.min())*255
    x = x.astype(np.uint8)
    y = (y - y.min())/(y.max() - y.min())*255
    y = y.astype(np.uint8)
    value = ssim_struct_caculation(x, y)

    return value

def calculate_best_match(num, opt, x1, x2, y1, y2, window_h_start, window_h_end, window_w_start, window_w_end, stride, oriw_loc, orih_loc, sensed_image, reference_image, img_name, sensed_ori, reference_ori):
    # sensed_img is vis
    conf = 0.3
    logvar_conf = 5.5

    box_height = y2 - y1
    box_width = x2 - x1
    h = window_h_end - window_h_start
    w = window_w_end - window_w_start

    result = np.zeros([int((h - box_height) / stride), int((w - box_width) / stride)])
    result_rev = np.zeros([int((h - box_height) / stride), int((w - box_width) / stride)])
    x = sensed_image[y1:y2, x1:x2]
    for i in range(int((h - box_height) / stride)):
        for j in range(int((w - box_width) / stride)):
            reference_h_start = window_h_start + i * stride
            reference_h_end = window_h_start + i * stride + box_height
            reference_w_start = window_w_start + j * stride
            reference_w_end = window_w_start + j * stride + box_width
            y = reference_image[reference_h_start:reference_h_end, reference_w_start:reference_w_end]

            result[i, j] = calculate_similarity(opt, x, y)
            result_rev[i, j] = calculate_similarity(opt, 255 - x, y)

    result = np.maximum(result,result_rev)

    similarity_var = result.var()
    opt.var.append(similarity_var)

    if(np.max(result) < conf or -np.log(similarity_var) > logvar_conf):
        max_location = np.where(result == np.max(result))
        max_location[0][0] = 0
        max_location[1][0] = 0
        match_h_start = y1
        match_h_end = y2
        match_w_start = x1
        match_w_end = x2
    else:
        max_location = np.where(result == np.max(result))
        match_h_start = window_h_start + max_location[0][0] * stride
        match_h_end = window_h_start + max_location[0][0] * stride + box_height
        match_w_start = window_w_start + max_location[1][0] * stride
        match_w_end = window_w_start + max_location[1][0] * stride + box_width

    return max_location, match_h_start, match_h_end, match_w_start, match_w_end

def MI_match(opt, sensed_pred,reference_image,sensed_image, img_name, reference_ori, sensed_ori):
    sensed_pred_match = copy.deepcopy(sensed_pred)
    for k in range(len(sensed_pred[0])):

        x1 = int(sensed_pred[0][k][0].item())
        x2 = int(sensed_pred[0][k][2].item())
        y1 = int(sensed_pred[0][k][1].item())
        y2 = int(sensed_pred[0][k][3].item())

        stride = 1
        window_h_max,window_w_max = reference_image.shape

        window_h_start,window_h_end,window_w_start,window_w_end, oriw_loc, orih_loc = box2windows(opt, x1,x2,y1,y2,window_h_max,window_w_max,stride)

        max_location, match_h_start, match_h_end, match_w_start, match_w_end = calculate_best_match(str(k), opt, x1,x2,y1,y2,window_h_start,window_h_end,window_w_start,window_w_end,stride,oriw_loc,orih_loc,sensed_image,reference_image, img_name, sensed_ori, reference_ori)

        sensed_pred_match[0][k][1] = match_h_start
        sensed_pred_match[0][k][0] = match_w_start
        sensed_pred_match[0][k][3] = match_h_end
        sensed_pred_match[0][k][2] = match_w_end

    return sensed_pred_match

def pred_NMS(pred1,pred2,iou_thres):        #pred1 is sensed(ir)
    pred_result = copy.deepcopy(pred2)
    fuse_iou = jaccard(pred1[0][:, :4], pred2[0][:, :4])
    m, n = fuse_iou.shape
    for i in range(m):
        add = True
        temp4 = 0
        for j in range(n):
            if (fuse_iou[i][j] > iou_thres):
                add = False
                if ((pred1[0][i][4] > pred2[0][j][4]) and (fuse_iou[i][j] > temp4)):
                    temp4 = fuse_iou[i][j]
                    pred_result[0][j][4:6] = pred1[0][i][4:6]
        if add == True:
            print('Fuse_Done_Add')
            temp5 = copy.deepcopy(pred1[0][i])
            pred_result[0] = torch.cat([pred_result[0], torch.unsqueeze(temp5, 0)], dim=0)

    return pred_result

def box_area(box):
    return (box[2]-box[0]) * (box[3]-box[1])

def pred_NMSC(pred, contain_thres, class_num):        #pred1 is sensed(ir)
    class_pred = [torch.zeros((0, 6), device=pred[0].device)] * class_num
    result = [torch.zeros((0, 6), device=pred[0].device)]
    for i in range(len(pred[0])):
        class_pred[int(pred[0][i][5])] = torch.cat((class_pred[int(pred[0][i][5])],torch.unsqueeze(pred[0][i],0)),0)

    for i in range(len(class_pred)):
        if len(class_pred[i]) >= 2:
            flag = True
            while(flag):
                last_pred = copy.deepcopy(class_pred[i])
                contain_iou = contain(class_pred[i][:, :4], class_pred[i][:, :4])
                for j in range(len(contain_iou)):
                    contain_iou[j][j] = 0
                    contain_numpy = contain_iou[j].cpu().numpy()
                    if np.max(contain_numpy) > contain_thres:
                        max_location = np.where(contain_numpy == np.max(contain_numpy))
                        if (box_area(class_pred[i][j]) > box_area(class_pred[i][max_location[0][0]])):
                            class_pred_cls_max = torch.max(class_pred[i][:,4])
                            for k in range(len(class_pred[i])):
                                class_pred[i][k][4] = class_pred_cls_max
                            class_pred[i] = class_pred[i][torch.arange(class_pred[i].size(0))!=max_location[0][0]]
                            break
                        else:
                            class_pred[i] = class_pred[i][torch.arange(class_pred[i].size(0)) != j]
                            break
                if(last_pred.equal(class_pred[i])):
                    break
    for i in range(len(class_pred)):
        result[0] = torch.cat((result[0], class_pred[i]), 0)

    return result

def Dec_Fusion_Multispectral_Withmatch(opt, img,img_ir,pred,pred_ir,iou_thres,contain_thres, img_name):
    print('/n')
    if len(pred[0]) == 0:
        pred_fuse = copy.deepcopy(pred_ir)
        for i in range(len(pred_ir[0])):
            pred_fuse[0][i][5] = 1

    else:
        img = img.cpu().numpy()[0,:, :, :]*255
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        r, g, b = cv2.split(img)
        img = cv2.merge([b, g, r])
        VIS_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_ir = img_ir.cpu().numpy()[0,:,:,:]*255
        img_ir = np.transpose(img_ir, (1, 2, 0))
        img_ir = img_ir.astype(np.uint8)
        r, g, b = cv2.split(img_ir)
        img_ir = cv2.merge([b, g, r])
        IR_grey = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)

        VIS_pro, IR_pro = img_prepro(opt, VIS_grey, IR_grey, img_name)
        pred_match = MI_match(opt, pred, IR_pro, VIS_pro, img_name, IR_grey, VIS_grey)
        pred_fuse = Dec_Fusion_Multispectral(pred_ir, pred_match, iou_thres, contain_thres)

    return pred_fuse
