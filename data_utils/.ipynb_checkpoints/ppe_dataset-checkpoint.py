import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import cv2
import einops
import os
from .mosaic import Mosaic
from torchvision.transforms import v2
import random
from torchvision import tv_tensors
class PPE_DATA(Dataset):
    def __init__(self, data_path: str = "./data", mode="train", 
                 max_ground_truth_boxes=30, p_mosaic = 1/32, apply_transforms=False, 
                 include_eyewear = True):
        # read file names from train.txt or validation.txt file
        self.mode = mode
        if mode == "train":
            self.file_names = np.genfromtxt(f"{data_path}/train.txt", dtype=str, delimiter="\n")
        elif mode == "val":
            self.file_names = np.genfromtxt(f"{data_path}/validation.txt", dtype=str, delimiter="\n")
        else:
            raise Exception("Invalid Mode Entered")
        
        self.max_gt = max_ground_truth_boxes
        self.p_mosaic = p_mosaic
        self.mosaic = Mosaic(include_eyewear = include_eyewear)

        # a boolean of whether or not to apply transforms
        self.transforms = apply_transforms
        self.include_eyewear = include_eyewear  # if false, will not include eyewear labels in the dataset

    def apply_transforms(self, img, labels):
        # This returns the new img and normalized labels in the form 
        # apply mosaic augmentation with a probability of p_mosaic
        if random.random() < self.p_mosaic:
            img = self.resize_img(img, output_size=640)
            img, labels = self.mosaic.forward(img, labels, self.file_names, output_size=640)
        else:
            img, labels = self.resize_and_pad_img(img, labels, output_size=640)
        
        # at this point it should be 640, if not something messed up with mosaic most likely
        img_size = 640
        cx, w, y, h = labels[:, 1:2] * img_size, labels[:, 3:4] * img_size, labels[:, 2:3] * img_size, labels[:, 4:5] * img_size
        x_min = (cx - w / 2)
        y_min = (y - h / 2)
        box_labels = torch.cat([x_min, y_min, w, h], dim=1)
        boxes = tv_tensors.BoundingBoxes(box_labels, format="XYWH", canvas_size=(img_size, img_size))
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
        img, boxes = transforms(img, boxes)
        labels = torch.cat([labels[:, :1], boxes], dim=1)  # keep class and confidence score
        labels[:,1] = labels[:, 1] + labels[:,3] / 2
        labels[:,2] = labels[:, 2] + labels[:,4] / 2
        labels[:,1:5] = labels[:, 1:5] / img_size
        return img, labels

    # Used for validaation, just resize and pad
    def resize_and_pad_img(self, img, labels,  output_size = 640):

        height, width = img.shape[1:]
        scale = min(output_size / width, output_size / height)
        if scale < 1:
            new_width, new_height = int(width * scale), int(height * scale)
        else:
            # don't scale up the image if it is already smaller than output size
            new_width, new_height = width, height
        img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        # pad with grey (114, 114, 114), not normalized
        pad_top = (output_size - new_height) // 2
        pad_bottom = output_size - new_height - pad_top
        pad_left = (output_size - new_width) // 2
        pad_right = output_size - new_width - pad_left        

        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value = 114.0)

        # scale labels up to pixel coords, scale by the same refactoring, add padding, then normalize
        if scale < 1:
            if labels.any():
                labels[..., 1] = (labels[..., 1] * width * scale + pad_left) / output_size   # xc
                labels[..., 2] = (labels[..., 2] * height * scale + pad_top ) / output_size   # yc
                labels[..., 3] = (labels[..., 3] * width * scale) / output_size              # w
                labels[..., 4] = (labels[..., 4] * height * scale) / output_size              # h
        else:
            if labels.any():
                labels[..., 1] = (labels[..., 1] * width + pad_left) / output_size   # xc
                labels[..., 2] = (labels[..., 2] * height + pad_top) / output_size   # yc
                labels[..., 3] = (labels[..., 3] * width) / output_size              # w
                labels[..., 4] = (labels[..., 4] * height) / output_size              # h

        return img.squeeze(0), labels
    
    # Used for training as input into transforms
    def resize_img(self, img, output_size = 640):
        # resize image to output_size, keeping aspect ratio
        height, width = img.shape[1:]
        scale = min(output_size / width, output_size / height)

        new_width, new_height = int(width * scale), int(height * scale)
        if new_width > width and new_height > height:
            return img  # no need to resize if img is smaller than output size

        img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        return img.squeeze(0)

    def read_img_and_labels(self, img_path):
        img = cv2.imread(img_path)

        lbl_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if os.stat(lbl_path).st_size == 0:
            labels = np.empty((0, 5), dtype=np.float32)  # empty tensor for no labels
        else:
            labels = np.loadtxt(lbl_path, dtype=np.float32)
            if labels.ndim == 1:
                labels = labels.reshape(1, -1)
        img = torch.from_numpy(img).float() #/ 255.0 # normalize the image
        img = einops.rearrange(img, "h w c -> c h w")
        labels = torch.from_numpy(labels)
        if not self.include_eyewear:
            mask = (labels[:, 0] != 2) & (labels[:, 0] != 3)
            labels = labels[mask]
        return img, labels

    def __getitem__(self, idx):
        img_path = self.file_names[idx]
        img, labels = self.read_img_and_labels(img_path)

        if self.transforms:
            img, labels = self.apply_transforms(img, labels)
        else:
            img, labels = self.resize_and_pad_img(img, labels, output_size=640)

        if labels.shape[0] > self.max_gt:
            raise Exception(f"Too many ground truth boxes in {img_path}: {labels.shape[0]} > {self.max_gt}")

        # convert labels to tensor, add padding if necessary
        padding = torch.ones(self.max_gt - labels.shape[0], 5)
        padding[:, 0] = -1  # set class to -1 for padding
        labels = torch.cat((labels, padding), dim=0)
        
        if self.mode == "val":
            # idx will be used as an img id for metric calculation
            return torch.tensor(idx), img, labels
        # for training, we return the image and labels
        return img, labels

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def show_img(img, labels, output_path=os.path.join("output_images", "output.png"), rect_coords_centered = True, 
                 normalized = True, show_conf_score = False):
        # Currently only accepts 3D pytorch tensors, outputs to img file
        # most comments are for the matplotlib code, switched to using opencv
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.shape[0] == 3:
            n = einops.rearrange(img, "c h w -> h w c").cpu().numpy().copy()
        else:
            n = img.cpu().numpy().copy()
        if n.dtype == np.float32 or n.dtype == np.float64:
            n = n.astype(np.uint8)

        # use this to draw different colors for each label
        # edge_colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]
        edge_colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (128,0,255), (255,128,0)]

        class_names = ["Ascomycetes/Ascospores", "Basidiomycetes/Basidiospores", "Cladosporium", "Hyphal fragments", "Non-specified Spore",
    "Penicillium/Aspergillus-like", "Rusts/Smuts/Periconia/Myxomycetes", "Debris",]
        #TODO does this always work?
        if labels.ndim == 3:
            labels = labels.squeeze(0)
            
        n = n.astype(np.uint8)
        for label in labels:
            if rect_coords_centered:
                c, x, y, w, h = label[:5]
                if c == -1:
                    continue
                if normalized:
                    x *= n.shape[1]
                    y *= n.shape[0]
                    w *= n.shape[1]
                    h *= n.shape[0]

                # yolo format gives x and y as center coordinates, convert to top-left corner
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
            else:
                c, x1, y1, x2, y2 = label[:5]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                if c == -1:
                    continue
            text = class_names[int(c.item())]
            if show_conf_score:
                s = float(label[5])
                text = f"{text} {s:.2f}"
            
            color = edge_colors[int(c.item())]

            # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=edge_color, facecolor='none')
            cv2.rectangle(n, (x1, y1), (x2, y2), color, 1)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.5, thickness=1)
            cv2.rectangle(n, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)   # filled bg
            cv2.putText(n, text, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        print(f"Saving image with predictions to {output_path}")
        cv2.imwrite(output_path, n)

if __name__ == "__main__":
    dataset = PPE_DATA(include_eyewear=False)
    index = int(random.random() * len(dataset))
    print(index)
    img, labels = dataset.__getitem__(index)
    PPE_DATA.show_img(img, labels, output_path="output_images/output.png")