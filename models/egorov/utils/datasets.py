import cv2

import numpy as np
import torch
import torch.utils.data



class MRIClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        self.transforms = transforms
        self.imgs = df['image_path'].tolist()
        self.masks = df['mask_path'].tolist()
        self.classes = df['tumor_class'].tolist()

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        tumor_class = self.classes[idx]

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)[:, :, 0]
        mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]
        mask[mask == 255] = 1

        pos = np.where(mask)

        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]])

        labels = torch.as_tensor(np.array([tumor_class]), dtype=torch.int64)
        masks = torch.as_tensor(np.array([mask]), dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1, ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)