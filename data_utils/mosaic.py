import torch
from torch import nn
import cv2
import einops
import torch.nn.functional as F
import numpy as np
import os
import random

class Mosaic(nn.Module):
    def __init__(self, excluded_classes = []):
        '''
        Initialize Mosaic augmentation module.

        :param excluded_classes: List of class IDs to exclude from labels. Defaults to an empty list.
        :type excluded_classes: list
        '''
        super(Mosaic, self).__init__()
        self.excluded_classes = excluded_classes

    def resize_and_pad_img(self, img: np.ndarray, labels: torch.Tensor, output_width: int, output_height: int) -> tuple:
        '''
        Resize and pad an image to the target dimensions, adjusting labels accordingly.

        :param img: Input image as a numpy array.
        :type img: np.ndarray
        :param labels: Corresponding labels for the image.
        :type labels: torch.Tensor
        :param output_width: Target width after resizing and padding.
        :type output_width: int
        :param output_height: Target height after resizing and padding.
        :type output_height: int
        :return: Tuple of (resized and padded image as torch.Tensor, adjusted labels as torch.Tensor).
        :rtype: tuple
        '''
        new_img = torch.from_numpy(img).float() 
        new_img = einops.rearrange(new_img, "h w c -> c h w")
        height, width = new_img.shape[1:]

        scale = min(output_width / width, output_height / height)

        scale = scale if scale <= 1 else 1

        new_width, new_height = int(width * scale), int(height * scale)

        # shrink down the img if it's smaller, otherwise just pad
        new_img = F.interpolate(new_img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        # pad with grey (114, 114, 114), not normalized
        pad_top = (output_height - new_height) // 2
        pad_bottom = output_height - new_height - pad_top
        pad_left = (output_width - new_width) // 2
        pad_right = output_width - new_width - pad_left

        new_img = F.pad(new_img, (pad_left, pad_right, pad_top, pad_bottom), value=114.0)

        # scale labels up to pixel coords, scale by the same refactoring, add padding, then normalize
        if labels.any():
            labels[..., 1] = (labels[..., 1] * width * scale + pad_left) / output_width   # xc
            labels[..., 2] = (labels[..., 2] * height * scale + pad_top) / output_height  # yc
            labels[..., 3] = (labels[..., 3] * width * scale) / output_width              # w
            labels[..., 4] = (labels[..., 4] * height * scale) / output_height             # h

        return new_img.squeeze(0), labels
    
    def read_img_and_labels(self, img_path: str) -> tuple:
        '''
        Read an image and its corresponding label file, filtering eyewear classes if needed.

        :param img_path: Path to the image file.
        :type img_path: str
        :return: Tuple of (image as np.ndarray, labels as np.ndarray).
        :rtype: tuple
        '''
        new_img = cv2.imread(img_path)
        lbl_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if not os.path.exists(lbl_path) or os.stat(lbl_path).st_size == 0:
            label = np.empty((0, 5), dtype=np.float32)  # empty tensor for no labels
        else:
            label = np.loadtxt(lbl_path, dtype=np.float32)
            if label.ndim == 1:
                label = label.reshape(1, -1)

        for excluded_cls in self.excluded_classes:
            mask = (label[:, 0] != excluded_cls)
            label = label[mask]

        return new_img, label

    def forward(self, img: torch.Tensor, labels: torch.Tensor, img_paths: list, output_size: int = 640) -> tuple:
        '''
        Apply mosaic augmentation to a batch of images and their corresponding labels.

        :param img: Main image tensor of shape (C, H, W), already resized to output size.
        :type img: torch.Tensor
        :param labels: Labels for the main image, shape (num_boxes, 5).
        :type labels: torch.Tensor
        :param img_paths: List of image file paths for the 3 random images to use in the mosaic.
        :type img_paths: list[str]
        :param output_size: Size to which the mosaic image will be cropped and padded. Defaults to 640.
        :type output_size: int
        :return: Tuple of (mosaic image as torch.Tensor, corresponding labels as torch.Tensor).
        :rtype: tuple
        '''
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
