import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.RBM.base import RBM
import models.RBM.transforms as transforms
from pycocotools.coco import COCO
from pathlib import Path
import numpy as np
import cv2
from demo import Demo, DemoTransform
from models.RBM.door_data import my_transforms


# https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/2
# class CorrectPartsDataset(Dataset):
#     def __init__(self, img_dir, annotation_file, transform=None):
#         self.img_dir = Path(img_dir)
#         self.transform = transform
#         self.annotation_file = Path(annotation_file)
#         # self._available_files = list(self.img_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff}'))
#         self._available_files = list(self.img_dir.glob('*.jpg'))
#
#         self.coco = COCO(self.annotation_file)
#         cats = self.coco.loadCats(self.coco.getCatIds())
#
#         self.anns = []
#
#         for name in [cat["name"] for cat in cats]:
#             catIds = self.coco.getCatIds(catNms=[name])
#             imgIds = self.coco.getImgIds(catIds=catIds)
#             annIds = self.coco.getAnnIds(catIds=catIds)
#             self.anns.extend(annIds)
#             print(f"{name} -- {len(imgIds)} -- {len(annIds)}")
#
#         # self.anns = self.coco.loadAnns(list(anns_by_cat))
#
#         # for part in self.parts:
#         # print(self.anns)
#         self.anns = self.coco.loadAnns(list(self.anns))
#
#     def __len__(self):
#         return len(self.anns)
#
#     def __getitem__(self, index):
#         mask = self.coco.annToMask(self.anns[index])
#         imgs = self.coco.loadImgs([self.anns[index]["image_id"]])[0]
#         img_path = Path((str(self.img_dir) + "/" + imgs["path"]))
#         img = cv2.imread(str(img_path.absolute()))
#
#         # Mask decomposition
#         first = img[:, :, 0]
#         second = img[:, :, 1]
#         third = img[:, :, 2]
#
#         cut_first = first * mask
#         cut_second = second * mask
#         cut_third = third * mask
#
#         img = np.dstack((cut_first, cut_second, cut_third))
#         # img_path = Path(self._available_files[index])
#         # print(str(img_path.absolute()))
#         # img = cv2.imread(str(img_path.absolute()))
#         if self.transform:
#             img = self.transform(img)
#         return img


# class BrokenPartsDataset(Dataset):
#     def __init__(self):
#         pass
#
#     def __len__(self):
#         pass
#
#     def __getitem__(self, index):
#         pass


