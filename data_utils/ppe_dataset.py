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
    '''
    Dataset for PPE detection in YOLO format with optional mosaic and torchvision v2 augmentations.

    This dataset reads newline-delimited image paths from a text file (e.g., ``train.txt``).
    For each image path under ``.../images/...``, a corresponding label file is expected at
    ``.../labels/<stem>.txt`` in YOLO format as rows of
    ``[class_id, x_center, y_center, width, height]`` normalized to ``[0, 1]``.

    :param data_path: Path to a text file containing image paths (one per line).
    :type data_path: str
    :param mode: Dataset mode; ``"train"`` or ``"val"``. In ``"val"``, ``__getitem__`` returns ``(idx, img, labels)``.
    :type mode: str
    :param max_gt: Maximum number of ground-truth boxes per image. Labels are padded (class ``-1``) to this length.
    :type max_gt: int
    :param p_mosaic: Probability of applying 4-image mosaic augmentation.
    :type p_mosaic: float
    :param apply_transforms: If ``True``, apply mosaic/flip/affine; otherwise only resize-and-pad to 640×640.
    :type apply_transforms: bool
    :param excluded_classes: Class IDs to ignore when sampling in mosaic.
    :type excluded_classes: list[int]

    :ivar file_names: Array of image paths loaded from ``data_path``.
    :vartype file_names: np.ndarray
    :ivar max_gt: See parameter description.
    :vartype max_gt: int
    :ivar p_mosaic: See parameter description.
    :vartype p_mosaic: float
    :ivar mosaic: Mosaic augmentation helper instance.
    :vartype mosaic: Mosaic
    :ivar transforms: Whether to apply training-time transforms in ``__getitem__``.
    :vartype transforms: bool
    :ivar excluded_classes: See parameter description.
    :vartype excluded_classes: list[int]
    '''
    def __init__(self, data_path: str = "./data/train.txt", mode="train", 
                 max_gt=200, p_mosaic = 1/32, apply_transforms=False, 
                 excluded_classes = []):
        '''
        Initialize the dataset and load image paths from ``data_path``.

        :param data_path: Path to newline-delimited image list (``.txt``).
        :type data_path: str
        :param mode: ``"train"`` or ``"val"``.
        :type mode: str
        :param max_gt: Maximum number of boxes per image (labels are padded to this length).
        :type max_gt: int
        :param p_mosaic: Probability to apply mosaic augmentation.
        :type p_mosaic: float
        :param apply_transforms: Apply augmentations if ``True``; otherwise only resize+pad.
        :type apply_transforms: bool
        :param excluded_classes: Class IDs to exclude when assembling mosaics.
        :type excluded_classes: list[int]
        :raises Exception: If ``data_path`` is not a valid file path.
        '''
        # read file names from train.txt or validation.txt file
        # data_path is either a path to your train/val directory or a direct path to a file 
        self.mode = mode
        if os.path.isfile(data_path):
            self.file_names = np.genfromtxt(f"{data_path}", dtype=str, delimiter="\n")
        else:
            raise Exception(f"data_path {data_path} is not a valid file path")
        
        self.max_gt = max_gt
        self.p_mosaic = p_mosaic
        self.mosaic = Mosaic(excluded_classes=excluded_classes)

        # a boolean of whether or not to apply transforms
        self.transforms = apply_transforms
        self.excluded_classes = excluded_classes

    def apply_transforms(self, img, labels):
        '''
        Apply training-time augmentations (mosaic with probability ``p_mosaic``,
        then geometric transforms) and keep labels normalized to the final canvas.

        Inputs are expected to be a 3xHxW float tensor in BGR channel order and
        YOLO-normalized labels of shape ``(N, 5)`` as
        ``[class_id, x_center, y_center, width, height]``.

        :param img: Image tensor before augmentation.
        :type img: torch.Tensor
        :param labels: Normalized YOLO labels for ``img``.
        :type labels: torch.Tensor
        :return: Tuple ``(img, labels)`` where ``img`` is 3x640x640 and labels remain normalized to ``[0, 1]``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        '''
        # This returns the new img and normalized labels in the form 
        # apply mosaic augmentation with a probability of p_mosaic
        if random.random() < self.p_mosaic:
            img = self.resize_img(img, output_size=640)
            img, labels = self.mosaic.forward(img, labels, self.file_names, output_size=640)
        else:
            img, labels = self.resize_and_pad_img_and_labels(img, labels, output_size=640)
        
        # at this point it should be 640, if not something messed up with mosaic most likely
        img_size = 640
        cx, y, w, h = labels[:, 1:2] * img_size,  labels[:, 2:3] * img_size, labels[:, 3:4] * img_size, labels[:, 4:5] * img_size
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
    @staticmethod
    def resize_and_pad_img_and_labels(img, labels,  output_size = 640):
        '''
        Resize (without upscaling) and pad an image to a square canvas, adjusting YOLO-normalized labels.

        The image is resized with preserved aspect ratio to fit within ``output_size x output_size``,
        then padded with gray (114,114,114). Labels are assumed to be YOLO-normalized to the **original**
        image dimensions and are transformed to remain normalized to the **new** canvas.

        :param img: Input image tensor of shape 3xHxW (float).
        :type img: torch.Tensor
        :param labels: YOLO-normalized labels as ``(N, 5)`` = ``[class, x_c, y_c, w, h]``. May be empty.
        :type labels: torch.Tensor
        :param output_size: Target square canvas size in pixels.
        :type output_size: int
        :return: Tuple ``(img_out, labels_out)`` where ``img_out`` is 3x``output_size``x``output_size`` and labels remain normalized.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        '''
        height, width = img.shape[1:]
        scale = min(output_size / width, output_size / height)

        # if the image is smaller than 640 x 640, don't scale up just pad
        scale = scale if scale <= 1 else 1

        new_width, new_height = int(width * scale), int(height * scale)
        img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        # pad with grey (114, 114, 114), not normalized
        pad_top = (output_size - new_height) // 2
        pad_bottom = output_size - new_height - pad_top
        pad_left = (output_size - new_width) // 2
        pad_right = output_size - new_width - pad_left        

        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value = 114.0)

        # scale labels up to pixel coords, scale by the same refactoring, add padding, then normalize
        # if labels and labels.any():
        if labels is not None and labels.shape[0] > 0:    
            labels[..., 1] = (labels[..., 1] * width * scale + pad_left) / output_size   # xc
            labels[..., 2] = (labels[..., 2] * height * scale + pad_top ) / output_size   # yc
            labels[..., 3] = (labels[..., 3] * width * scale) / output_size              # w
            labels[..., 4] = (labels[..., 4] * height * scale) / output_size              # h

        return img.squeeze(0), labels
    
    # Used for training as input into transforms
    @staticmethod
    def resize_img(img, output_size = 640):
        '''
        Resize an image to fit within ``output_size x output_size`` while preserving aspect ratio.

        If the image is already smaller than ``output_size`` along both dimensions,
        no upscaling is performed and the original tensor is returned.

        :param img: Input image tensor of shape 3xHxW (float).
        :type img: torch.Tensor
        :param output_size: Maximum size of the longer side in pixels.
        :type output_size: int
        :return: Resized image tensor (no padding), shape 3xH'xW'.
        :rtype: torch.Tensor
        '''
        # resize image to output_size, keeping aspect ratio
        height, width = img.shape[1:]
        scale = min(output_size / width, output_size / height)

        new_width, new_height = int(width * scale), int(height * scale)
        if new_width > width and new_height > height:
            return img  # no need to resize if img is smaller than output size

        img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        return img.squeeze(0)

    def read_img_and_labels(self, img_path):
        '''
        Read an image (BGR) and its YOLO label file, returning tensors.

        The label file is expected at ``img_path`` with ``/images/`` replaced by ``/labels/``
        and the extension changed to ``.txt``. If the label file is missing or empty,
        an empty ``(0, 5)`` label tensor is returned.

        :param img_path: Path to the image file under an ``images`` directory.
        :type img_path: str
        :return: Tuple ``(img, labels)`` where ``img`` is a float tensor 3×H×W (BGR order) and
                 ``labels`` is a float tensor of shape ``(N, 5)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises FileNotFoundError: If the image cannot be loaded.
        '''
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}. \
                                    Maybe you forgot to prepend the directory name?")

        lbl_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if not os.path.exists(lbl_path) or os.stat(lbl_path).st_size == 0:
            labels = np.empty((0, 5), dtype=np.float32)  # empty tensor for no labels
        else:
            labels = np.loadtxt(lbl_path, dtype=np.float32)
            if labels.ndim == 1:
                labels = labels.reshape(1, -1)
        img = torch.from_numpy(img).float() #/ 255.0 # normalize the image
        img = einops.rearrange(img, "h w c -> c h w")
        labels = torch.from_numpy(labels)
        return img, labels

    def __getitem__(self, idx):
        '''
        Load and (optionally) augment a sample.

        In training mode, returns ``(img, labels)``; in validation mode, returns
        ``(idx, img, labels)``. Labels are padded to ``max_gt`` rows with class ``-1``.

        :param idx: Dataset index.
        :type idx: int
        :return: Depending on ``mode``, either ``(img, labels)`` or ``(idx, img, labels)``.
        :rtype: tuple
        :raises Exception: If the number of boxes exceeds ``max_gt``.
        '''
        img_path = self.file_names[idx]
        img, labels = self.read_img_and_labels(img_path)

        if self.transforms:
            img, labels = self.apply_transforms(img, labels)
        else:
            img, labels = self.resize_and_pad_img_and_labels(img, labels, output_size=640)

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
        '''
        Render bounding boxes and class names onto an image and write it to disk.

        By default, labels are interpreted as YOLO center-format boxes (``rect_coords_centered=True``).
        If ``normalized=True``, coordinates are treated as fractions of image width/height; otherwise
        they are pixel values. When ``rect_coords_centered=False``, labels are interpreted as
        ``[class, x1, y1, x2, y2]`` corner format.

        :param img: Image tensor of shape 3×H×W or 1×3×H×W.
        :type img: torch.Tensor
        :param labels: Labels tensor of shape ``(N, 5)`` (optionally with confidence at index 5).
        :type labels: torch.Tensor
        :param output_path: Destination filepath for the rendered image.
        :type output_path: str
        :param rect_coords_centered: Treat boxes as YOLO center format if ``True``; else corner format.
        :type rect_coords_centered: bool
        :param normalized: Interpret coordinates as normalized in ``[0, 1]`` if ``True``; else pixels.
        :type normalized: bool
        :param show_conf_score: If ``True``, append confidence (``labels[:, 5]``) to the class name if present.
        :type show_conf_score: bool
        :return: None
        :rtype: None
        '''
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
    dataset = PPE_DATA()
    data = "data/images/train/eurofins17_0148.jpg"
    img, labels = dataset.read_img_and_labels(data)
    PPE_DATA.show_img(img, labels)
