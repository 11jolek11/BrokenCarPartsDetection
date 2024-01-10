import os.path
import uuid
from pathlib import Path

from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .transforms import *


class DoorsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        if Path(img_dir).is_dir():
            self.img_dir = Path(img_dir)
        else:
            raise AttributeError("Directory not found")
        self.transform = transform
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

door_transforms2 = v2.Compose([
    # v2.ToImage(),
    # v2.Resize([32, 32]),  # FIXME(11jolek11): Is not resizing to 32x32
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True)
])


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


my_transforms_only_image = v2.Compose([
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
])

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

        # blur = cv.bilateralFilter(img,9,75,75)
        # https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html

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

        # print(binarized.shape)
        return binarized


class DoorsDataset3(Dataset):
    def __init__(self, img_dir, annotation_file, parts=None, transform=None):
        if parts is None:
            parts = ["hood"]
        self.parts = parts
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.annotation_file = Path(annotation_file)
        # self._available_files = list(self.img_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff}'))
        self._available_files = list(self.img_dir.glob('*.jpg'))

        self.coco = COCO(self.annotation_file)

        # Available categories
        # cats = self.coco.loadCats(self.coco.getCatIds())
        # print([cat["name"] for cat in cats])

        # ['_background_', 'back_bumper', 'back_glass', 'back_left_door', 'back_left_light', 'back_right_door',
        # 'back_right_light', 'front_bumper', 'front_glass', 'front_left_door', 'front_left_light', 'front_right_door',
        # 'front_right_light', 'hood', 'left_mirror', 'right_mirror', 'tailgate', 'trunk', 'wheel']

        # catIds = self.coco.getCatIds(catNms=["front_left_door"])
        catIds = self.coco.getCatIds(catNms=parts)
        imgIds = self.coco.getImgIds(catIds=catIds)

        self.imgs = self.coco.loadImgs(imgIds)

        annIds = self.coco.getAnnIds(imgIds=[img["id"] for img in self.imgs], catIds=catIds, iscrowd=None)
        self.anns = self.coco.loadAnns(annIds)

    def __len__(self):
        return len(self.anns)

    def get_parts(self):
        return self.parts

    def __getitem__(self, index):
        mask = self.coco.annToMask(self.anns[index])
        imgs = self.coco.loadImgs([self.anns[index]["image_id"]])[0]
        img_path = Path((str(self.img_dir) + "/" + imgs["path"]))
        # print(str(img_path.absolute()))
        img = cv2.imread(str(img_path.absolute()))

        # Mask decomposition
        first = img[:, :, 0]
        second = img[:, :, 1]
        third = img[:, :, 2]

        cut_first = first * mask
        cut_second = second * mask
        cut_third = third * mask

        img = np.dstack((cut_first, cut_second, cut_third))

        # img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
        img_blur = cv2.GaussianBlur(img, (13, 13), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=50)

        MIN_CONTOUR_AREA = 100
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

        return binarized


class DoorsDatasetSaver(Dataset):
    def __init__(self, img_dir, annotation_file, parts=None, transform=None):
        if parts is None:
            parts = ["hood"]
        self.annotation_file = Path(annotation_file)
        self.parts = parts
        self.img_dir = Path(img_dir)
        self.transform = transform
        # self._available_files = list(self.img_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff}'))
        self._available_files = list(self.img_dir.glob('*.jpg'))

        self.coco = COCO(self.annotation_file)

        # Available categories
        # cats = self.coco.loadCats(self.coco.getCatIds())
        # print([cat["name"] for cat in cats])

        # ['_background_', 'back_bumper', 'back_glass', 'back_left_door', 'back_left_light', 'back_right_door',
        # 'back_right_light', 'front_bumper', 'front_glass', 'front_left_door', 'front_left_light', 'front_right_door',
        # 'front_right_light', 'hood', 'left_mirror', 'right_mirror', 'tailgate', 'trunk', 'wheel']

        # catIds = self.coco.getCatIds(catNms=["front_left_door"])
        catIds = self.coco.getCatIds(catNms=parts)
        imgIds = self.coco.getImgIds(catIds=catIds)

        self.imgs = self.coco.loadImgs(imgIds)

        annIds = self.coco.getAnnIds(imgIds=[img["id"] for img in self.imgs], catIds=catIds, iscrowd=None)
        self.anns = self.coco.loadAnns(annIds)

    def __len__(self):
        return len(self.anns)

    def get_parts(self):
        return self.parts

    def __getitem__(self, index):
        mask = self.coco.annToMask(self.anns[index])
        imgs = self.coco.loadImgs([self.anns[index]["image_id"]])[0]
        img_path = Path((str(self.img_dir) + "/" + imgs["path"]))
        # print(str(img_path.absolute()))
        img = cv2.imread(str(img_path.absolute()))

        # Mask decomposition
        first = img[:, :, 0]
        second = img[:, :, 1]
        third = img[:, :, 2]

        cut_first = first * mask
        cut_second = second * mask
        cut_third = third * mask

        img = np.dstack((cut_first, cut_second, cut_third))

        # img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
        img_blur = cv2.GaussianBlur(img, (13, 13), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=50)

        MIN_CONTOUR_AREA = 100
        img_thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        Contours, imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in Contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                [X, Y, W, H] = cv2.boundingRect(contour)
                cropped_image = edges[Y:Y + H, X:X + W]



        resized_edges = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_AREA)

        img_uuid = str(uuid.uuid4())
        temp_path = f"C:\\Users\\dabro\\PycharmProjects\\scientificProject\\notebooks\\CarPartsDatasetExperimentDir\\exp2\\{img_uuid}.jpg"
        cv2.imwrite(temp_path, resized_edges)
        if os.path.exists(temp_path):
            print(f"{img_uuid} created")

        binarized = np.where(resized_edges > 100, 1.0, 0.0)

        if self.transform:
            binarized = self.transform(binarized)

        return binarized


class CarDataset(Dataset):
    def __init__(self, img_dir, annotation_file, parts=None, transform=None):
        if parts is None:
            parts = ["hood"]
        self.parts = parts
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.annotation_file = Path(annotation_file)
        # self._available_files = list(self.img_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff}'))
        self._available_files = list(self.img_dir.glob('*.jpg'))

        self.coco = COCO(self.annotation_file)

        # Available categories
        # cats = self.coco.loadCats(self.coco.getCatIds())
        # print([cat["name"] for cat in cats])

        # ['_background_', 'back_bumper', 'back_glass', 'back_left_door', 'back_left_light', 'back_right_door',
        # 'back_right_light', 'front_bumper', 'front_glass', 'front_left_door', 'front_left_light', 'front_right_door',
        # 'front_right_light', 'hood', 'left_mirror', 'right_mirror', 'tailgate', 'trunk', 'wheel']

        # "front_left_door",'front_right_door', 'hood',"front_bumper" , 'back_bumper', 'tailgate'

        # catIds = self.coco.getCatIds(catNms=["front_left_door"])

        cats = self.coco.loadCats(self.coco.getCatIds())

        self.experiment_dict = dict()

        for name in [cat["name"] for cat in cats]:
            catIds = self.coco.getCatIds(catNms=[name])
            imgIds = self.coco.getImgIds(catIds=catIds)
            annIds = self.coco.getAnnIds(catIds=catIds)
            self.experiment_dict[name] = annIds
            print(f"{name} -- {len(imgIds)} -- {len(annIds)}")

        # self.anns = self.coco.loadAnns(list(anns_by_cat))
        self.anns = []

        for part in self.parts:
            self.anns.extend(self.experiment_dict[part])
        # print(self.anns)
        self.anns = self.coco.loadAnns(list(self.anns))

    def __len__(self):
        return len(self.anns)

    def get_parts(self):
        return self.parts

    def __getitem__(self, index):
        mask = self.coco.annToMask(self.anns[index])
        imgs = self.coco.loadImgs([self.anns[index]["image_id"]])[0]
        img_path = Path((str(self.img_dir) + "/" + imgs["path"]))
        img = cv2.imread(str(img_path.absolute()))

        # Mask decomposition
        first = img[:, :, 0]
        second = img[:, :, 1]
        third = img[:, :, 2]

        cut_first = first * mask
        cut_second = second * mask
        cut_third = third * mask

        img = np.dstack((cut_first, cut_second, cut_third))
        # img_path = Path(self._available_files[index])
        # print(str(img_path.absolute()))
        # img = cv2.imread(str(img_path.absolute()))
        if self.transform:
            img = self.transform(img)
        return img


train_transforms = v2.Compose([
        GaussianBlur((13, 13)),
        BilateralFilter(30, 75, 75),
        Erode((9, 9)),
        Canny((50, 50)),
        GaussianBlur((3, 3)),
        FindBoundingBoxAndCrop(),
        Resize((128, 128)),
        Binarize(),
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])

train_transforms32 = v2.Compose([
        GaussianBlur((13, 13)),
        BilateralFilter(5, 15, 15),
        Erode((9, 9)),
        Canny((50, 50)),
        GaussianBlur((3, 3)),
        FindBoundingBoxAndCrop(),
        Resize((32, 32)),
        Binarize(),
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])

if __name__ == "__main__":
    test_data = CarDataset(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
        parts=["hood", "wheel"]
    )
    print(f'test data size {len(test_data)}')

    # test_loader = DataLoader(dataset=datas_test, batch_size=1, shuffle=True)
    # for image in test_loader:
    #     img = v2.ToPILImage()(image)
    #     img.show()
