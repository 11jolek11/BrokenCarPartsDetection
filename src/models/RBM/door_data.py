import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision.transforms import v2
import cv2
from torchvision.io import read_image as torch_read_image
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
import matplotlib.image as mpimg


class DoorsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        if Path(img_dir).is_dir():
            self.img_dir = Path(img_dir)
        else:
            raise AttributeError("Directory not found")
        self.transform = transform
        # self._available_files = list(self.img_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff}'))
        self._available_files = list(self.img_dir.glob('*.jpg'))

    def __len__(self):
        return len(self._available_files)

    def __getitem__(self, index):
        img_path = self._available_files[index]
        img = cv2.imread(str(img_path.absolute()))
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        img = np.where(img > 100, 1.0, 0.0)

        if self.transform:
            img = self.transform(img)

        return img


door_transforms = v2.Compose([
    # v2.ToImage(),
    # v2.Resize([32, 32]),  # FIXME(11jolek11): Is not resizing to 32x32
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True)
])



class Canny(object):
    pass


class Binarize(object):
    pass

class DoorsDataset2(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        self.annotation_file = Path(annotation_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        # self._available_files = list(self.img_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff}'))
        self._available_files = list(self.img_dir.glob('*.jpg'))

        self.coco = COCO(self.annotation_file)

        catIds = self.coco.getCatIds(catNms=["front_left_door"])
        imgIds = self.coco.getImgIds(catIds=catIds)

        imgs = self.coco.loadImgs(imgIds)

        annIds = self.coco.getAnnIds(imgIds=[img["id"] for img in imgs], catIds=catIds, iscrowd=None)
        self.anns = self.coco.loadAnns(annIds)

        # for i in range(len(anns)):
        #     mask = coco.annToMask(anns[i])
        #     img_p = mpimg.imread(f"{data_dir}/{imgs[i]['path']}")
        #
        #     # Mask decomposition
        #     first = img_p[:, :, 0]
        #     second = img_p[:, :, 1]
        #     third = img_p[:, :, 2]
        #
        #     cut_first = first * mask
        #     cut_second = second * mask
        #     cut_third = third * mask
        #
        #     cut = np.dstack((cut_first, cut_second, cut_third))
            # all_cuts.append((cut, f"{data_dir}/{imgs[i]['path']}"))

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        # print(len(self.anns))
        # print(len(self._available_files))
        mask = self.coco.annToMask(self.anns[index])
        img_path = self._available_files[index]
        img = cv2.imread(str(img_path.absolute()))

        # Mask decomposition
        first = img[:, :, 0]
        second = img[:, :, 1]
        third = img[:, :, 2]

        cut_first = first * mask
        cut_second = second * mask
        cut_third = third * mask

        img = np.dstack((cut_first, cut_second, cut_third))

        img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=50)

        MIN_CONTOUR_AREA = 1000
        img_thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        Contours, imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in Contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                [X, Y, W, H] = cv2.boundingRect(contour)
                cropped_image = edges[Y:Y + H, X:X + W]



        resized_edges = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_AREA)

        binarized = np.where(resized_edges > 100, 1.0, 0.0)

        if self.transform:
            binarized = self.transform(binarized)

        print(binarized.shape)
        return binarized

if __name__ == "__main__":
    b = DoorsDataset2(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/JPEGImages",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json"
    )

    image = b[0]
    print(np.unique(image))

    # cv2.imshow("test", image)
    # cv2.waitKey(0)
