import torch
from torch import nn
import cv2
import einops
import torch.nn.functional as F
import numpy as np
import os
import random

class Mosaic(nn.Module):
    def __init__(self, include_eyewear = True):
        super(Mosaic, self).__init__()
        self.include_eyewear = include_eyewear
        
    def resize_and_pad_img(self, img, labels, output_width, output_height):
        
        new_img = torch.from_numpy(img).float() 
        new_img = einops.rearrange(new_img, "h w c -> c h w")
        height, width = new_img.shape[1:]

        scale = min(output_width / width, output_height / height)

        if scale < 1:
            new_width, new_height = int(width * scale), int(height * scale)
        else:
            new_width, new_height = width, height

        # shrink down the img if it's smaller, otherwise just pad
        new_img = F.interpolate(new_img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        # pad with grey (114, 114, 114), not normalized
        pad_top = (output_height - new_height) // 2
        pad_bottom = output_height - new_height - pad_top
        pad_left = (output_width - new_width) // 2
        pad_right = output_width - new_width - pad_left

        new_img = F.pad(new_img, (pad_left, pad_right, pad_top, pad_bottom), value=114.0)

        # scale labels up to pixel coords, scale by the same refactoring, add padding, then normalize
        if scale < 1:
            if labels.any():
                labels[..., 1] = (labels[..., 1] * width * scale + pad_left) / output_width   # xc
                labels[..., 2] = (labels[..., 2] * height * scale + pad_top) / output_height  # yc
                labels[..., 3] = (labels[..., 3] * width * scale) / output_width              # w
                labels[..., 4] = (labels[..., 4] * height * scale) / output_height             # h
        else:
            if labels.any():
                labels[..., 1] = (labels[..., 1] * width + pad_left) / output_width   # xc
                labels[..., 2] = (labels[..., 2] * height + pad_top) / output_height  # yc
                labels[..., 3] = (labels[..., 3] * width) / output_width              # w
                labels[..., 4] = (labels[..., 4] * height) / output_height             # h

        return new_img.squeeze(0), labels
    
    def read_img_and_labels(self, img_path):

        new_img = cv2.imread(img_path)
        lbl_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if os.stat(lbl_path).st_size == 0:
            label = np.empty((0, 5), dtype=np.float32)  # empty tensor for no labels
        else:
            label = np.loadtxt(lbl_path, dtype=np.float32)
            if label.ndim == 1:
                label = label.reshape(1, -1)

        if not self.include_eyewear:
            mask = (label[:, 0] != 2) & (label[:, 0] != 3)
            label = label[mask]

        return new_img, label

    def forward(self, img, labels, img_paths, output_size=640):
        """
        Apply mosaic augmentation to a batch of images and their corresponding labels.
        
        Args:
            img (torch.Tensor): Image in shape (C, H, W). img should already be resized to the desired output size.
            img_paths (list of str): List of image file paths corresponding to the 3 random 
            images to be used in the mosaic. The paths should point to the original images, not to the resized
            imgs (list of torch.Tensor): List of images in shape (C, H, W) to be used in the mosaic.
            labels (list of torch.Tensor): List of labels in shape (num_boxes, 4).
            Formatted in (cx, cy, w, h) where cx, cy are normalized center coordinates and w, h are normalized width and height.
            output_size (int): Size to which the mosaic image will be resized.
        
        Returns:
            torch.Tensor: Mosaic image.
            torch.Tensor: Corresponding labels for the mosaic image.
        """

        # Ensure input is a list of images and labels

        if type(labels) == np.ndarray:        
            labels = torch.from_numpy(labels)
        indices = torch.randperm(len(img_paths))[:3]
        imgs = []
        lbl_list = []
        height, width = img.shape[1:]

        # images can be of variable size, so we will make them match the input img size
        for i in indices:
            new_img, label = self.read_img_and_labels(img_paths[i])
            new_img, label = self.resize_and_pad_img(new_img, label, output_width=width, output_height=height)
            
            assert(new_img.shape[1:] == img.shape[1:])
            imgs.append(new_img.squeeze(0))

            lbl_list.append(torch.from_numpy(label))

        top = torch.cat((img, imgs[0]), dim=2)  # Concatenate horizontally
        bottom = torch.cat((imgs[1], imgs[2]), dim=2)
        imgs = torch.cat((top, bottom), dim=1)  # Concatenate vertically

        total_width = imgs.shape[2]
        total_height = imgs.shape[1]
        w_index = random.randint(0, max(total_width - output_size, 0))
        h_index = random.randint(0, max(total_height - output_size, 0))
        cropped_img = imgs[:, h_index:h_index + output_size, w_index:w_index+output_size]
        pad_right = max(0, output_size - cropped_img.shape[2])
        pad_bottom = max(0, output_size - cropped_img.shape[1])
        cropped_img = F.pad(cropped_img, (0, pad_right, 0, pad_bottom), value=114.0)

        fin_labels = torch.empty((0, 5), dtype=torch.float32)

        # Top left corner:
        cx, cy, cw, ch = labels[:, 1:2] * width, labels[:, 2:3] * height, labels[:, 3:4] * width, labels[:, 4:5] * height 

        labels[:, 1:2] = cx - w_index
        labels[:, 2:3] = cy - h_index
        labels[:, 3:4] = cw
        labels[:, 4:5] = ch
        mask = ((labels[:, 1] + labels[:, 3] / 2) > 0) & ((labels[:, 2] + labels[:, 4] / 2) > 0)
        labels = labels[mask]
        labels[:, 1] = labels[:, 1] / output_size
        labels[:, 2] = labels[:, 2] / output_size
        labels[:, 3] = labels[:, 3] / output_size
        labels[:, 4] = labels[:, 4] / output_size
        fin_labels = torch.cat((fin_labels, labels), dim=0)

        # # Top right corner:
        # # TODO how could this be generalized
        labels = lbl_list[0].clone()
        cx, cy, cw, ch = labels[:, 1:2] * width, labels[:, 2:3] * height, labels[:, 3:4] * width, labels[:, 4:5] * height
        labels[:, 1:2] = cx + width - w_index
        labels[:, 2:3] = cy - h_index
        labels[:, 3:4] = cw
        labels[:, 4:5] = ch

        mask = ((labels[:, 1] - labels[:, 3] / 2) < output_size) & ((labels[:, 2] + labels[:, 4] / 2) > 0)
        labels = labels[mask]
        labels[:, 1] = labels[:, 1] / output_size
        labels[:, 2] = labels[:, 2] / output_size
        labels[:, 3] = labels[:, 3] / output_size
        labels[:, 4] = labels[:, 4] / output_size

        fin_labels = torch.cat((fin_labels, labels), dim=0)

        labels = lbl_list[1].clone()

        # # Bottom left corner:
        cx, cy, cw, ch = labels[:, 1:2] * width, labels[:, 2:3] * height, labels[:, 3:4] * width, labels[:, 4:5] * height
        labels[:, 1:2] = cx - w_index
        labels[:, 2:3] = cy + height - h_index
        labels[:, 3:4] = cw
        labels[:, 4:5] = ch
        mask = ((labels[:, 1] + labels[:, 3] / 2) > 0) & ((labels[:, 2] - labels[:, 4] / 2) < output_size)
        labels = labels[mask]
        labels[:, 1] = labels[:, 1] / output_size
        labels[:, 2] = labels[:, 2] / output_size
        labels[:, 3] = labels[:, 3] / output_size
        labels[:, 4] = labels[:, 4] / output_size
        fin_labels = torch.cat((fin_labels, labels), dim=0)

        # # Bottom right corner:
        labels = lbl_list[2].clone()
        cx, cy, cw, ch = labels[:, 1:2] * width, labels[:, 2:3] * height, labels[:, 3:4] * width, labels[:, 4:5] * height
        labels[:, 1:2] = cx + width - w_index
        labels[:, 2:3] = cy + height - h_index
        labels[:, 3:4] = cw
        labels[:, 4:5] = ch
        mask = ((labels[:, 1] - labels[:, 3] / 2) < output_size) & ((labels[:, 2] - labels[:, 4] / 2) < output_size)
        labels = labels[mask]
        labels[:, 1] = labels[:, 1] / output_size
        labels[:, 2] = labels[:, 2] / output_size
        labels[:, 3] = labels[:, 3] / output_size
        labels[:, 4] = labels[:, 4] / output_size
        fin_labels = torch.cat((fin_labels, labels), dim=0)

        return cropped_img, fin_labels
