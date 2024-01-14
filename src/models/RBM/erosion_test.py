# README FIRST !!!
# Kod testowy
from torchvision.transforms import v2
from transforms import GaussianBlur, BilateralFilter, Erode, Canny, FindBoundingBoxAndCrop, Resize, Binarize, ColorToHSV, RemoveInnerContours
import torch
from door_data import CarDataset
from torchvision.utils import make_grid
import torchvision.transforms as T


clear_transforms = v2.Compose([
    GaussianBlur((13, 13)),
    Canny((50, 50)),
    GaussianBlur((3, 3)),
    FindBoundingBoxAndCrop(),
    Resize((128, 128)),
    Binarize(),
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True)
])

# my_transforms = v2.Compose([
#     v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0)),
#     ColorToHSV(),
#     GaussianBlur((13, 13)),
#     Canny((50, 50)),
#     BilateralFilter(15, 75, 75),
#     Erode((15, 15)),
#     # GaussianBlur((3, 3)),
#     FindBoundingBoxAndCrop(),
#     Resize((128, 128)),
#     Binarize(),
#     v2.ToTensor(),
#     v2.ToDtype(torch.float32, scale=True)
# ])

my_transforms = v2.Compose([
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0)),
    ColorToHSV(),
    GaussianBlur((13, 13)),
    BilateralFilter(15, 75, 75),
    RemoveInnerContours(),
    # Erode((30, 30)),
    # GaussianBlur((3, 3)),
    FindBoundingBoxAndCrop(),
    Resize((128, 128)),
    Binarize(),
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True)
])

part = ["front_left_door", 'front_right_door', 'hood', 'back_glass', 'front_glass']

# part = ["front_left_door"]

reference = CarDataset(
    "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
    "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
    parts=part,
    transform=clear_transforms)

transformed = CarDataset(
    "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
    "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
    parts=part,
    transform=my_transforms)

img_list = []

for i in range(len(transformed)):
    img_list.append(reference[i])
    img_list.append(transformed[i])

grid = make_grid(img_list, nrow=2)
img = T.ToPILImage()(grid)
img.show()
