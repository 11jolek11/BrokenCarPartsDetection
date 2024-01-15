import copy
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset, random_split
from win11toast import notify

from .models.RBM.base import RBM
from .models.RBM.utildata.door_data import my_transforms
from .models.blocks import ReconstructionModel, SegmentationModel
from .models.RBM.settings import DEVICE

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from torch.utils.data import Dataset
import torch
from typing import Literal


class DemoTransform(Dataset):
    def __init__(self, frame, masks, legend: dict[str, np.ndarray], transform=None, target_size=(512, 512)):
        self.frame = frame
        self.masks = masks
        self.legend = legend
        self.target_size = target_size
        self.masks_selection = tuple(self.legend.keys())
        self.transform = transform

    def __len__(self):
        return len(self.masks_selection)

    def __getitem__(self, index):
        if self.frame.size != self.masks[index].size:
            frame = cv2.resize(self.frame, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            frame = self.frame

        mask = self.masks[:, :, self.legend[self.masks_selection[index]]]
        cut_first = frame[:, :, 0] * mask
        cut_second = frame[:, :, 1] * mask
        cut_third = frame[:, :, 2] * mask

        img = np.dstack((cut_first, cut_second, cut_third))

        img_copy = copy.deepcopy(img)

        img = (img * 255).astype(np.uint8)

        if self.transform:
            img = self.transform(img)

        return img, self.masks_selection[index], img_copy


class Demo:
    def __init__(self, target_size=(512, 512),
                 rbm_model_path: str = "model_zoo/RBM/RBM_cuda_12_28_2023_14_41_54_uuid_f509f783.pth",
                 unet_model_path: str = "model_zoo/UNet/unet_colab_t4_kxB53MMd.h5"):
        self.recon_model_path = Path(rbm_model_path)
        self.segmentation_model_path = Path(unet_model_path)

        # ['_background_', 'back_bumper', 'back_glass', 'back_left_door', 'back_left_light', 'back_right_door',
        # 'back_right_light', 'front_bumper', 'front_glass', 'front_left_door', 'front_left_light', 'front_right_door',
        # 'front_right_light', 'hood', 'left_mirror', 'right_mirror', 'tailgate', 'trunk', 'wheel']
        self.class_names = ['_background_', 'back_bumper', 'back_glass', 'back_left_door', 'back_left_light',
                            'back_right_door',
                            'back_right_light', 'front_bumper', 'front_glass', 'front_left_door', 'front_left_light',
                            'front_right_door',
                            'front_right_light', 'hood', 'left_mirror', 'right_mirror', 'tailgate', 'trunk', 'wheel']

        self.seg_model = sm.Unet
        self.recon_model = RBM

        self.target_size = target_size
        # (128 * 128, 128*128, k=3)

        self.seg_block = SegmentationModel(self.class_names, self.seg_model, self.segmentation_model_path,
                                           'resnet18', classes=len(self.class_names), activation='softmax')

        self.recon_block = ReconstructionModel(RBM, self.recon_model_path, 128 * 128, 128 * 128, k=3)

    def forward(self, frame: np.ndarray, to_image: bool = True):
        original_frames, mask_p, legends = self.seg_block.generate_masks(frame)
        data = DemoTransform(frame, mask_p, legends, transform=my_transforms, target_size=self.target_size)

        reconstructed_parts = dict()

        cutoffs = []

        for part_no in range(len(data)):
            data_temp, part_name, mask = data[part_no]

            resized_frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

            cut_first = resized_frame[:, :, 0] * mask_p[:, :, part_no]
            cut_second = resized_frame[:, :, 1] * mask_p[:, :, part_no]
            cut_third = resized_frame[:, :, 2] * mask_p[:, :, part_no]

            cutoff = np.dstack((cut_first, cut_second, cut_third))

            cutoffs.append(cutoff)

            _, temp = self.recon_block.reconstruct(data_temp, part_name)

            data_temp = v2.ToPILImage()(data_temp)

            to_return = dict()

            to_return[list(temp.keys())[0]] = {
                "recon": temp[list(temp.keys())[0]],
                "cutoff": cutoff,
                "transformed_part": data_temp
            }

            reconstructed_parts.update(to_return)

        return frame, reconstructed_parts


class DisHandle:
    def __init__(self, model_path: str = "./model_zoo/Discr/dis_model.pth") -> None:
        self.model_path = Path(model_path)
        self.model = Dis(2)

        temp = torch.load(str(self.model_path.absolute()))

        self.mapper = temp["map"]

        self.model.load_state_dict(temp["model"])

    def classify(self, diff: list[str, int]) -> float:
        diff[0] = self.mapper[diff[0]][0].item()
        # FIXME(11jolek11): TypeError: linear(): argument 'input' (position 1) must be Tensor, not list
        diff = torch.FloatTensor(diff)
        return self.model(diff).item()


class DisDataset(Dataset):
    def __init__(self, data_path: str | Path, supplier: Demo, data_type: Literal[0, 1], transform=None):
        if not Path(data_path).is_dir():
            raise ValueError("Expected Path to a existing dir")

        self.data_path = Path(data_path)
        self.supplier = supplier
        self.data_type = data_type
        self.transform = transform

        self._available_files = list(self.data_path.glob('*.jpg'))

        # TODO(11jolek11): Change this garbage code! ASAP
        self.mapper = dict()

        self.frame_collection = []
        self.recon_collection = []

        self.caster = v2.ToDtype(torch.float32, scale=False)

        self.data = []
        self.target = []

        self.unique_parts = set()



        frame_counter = 0
        recon_counter = 0

        for image_path in self._available_files:
            image = cv2.imread(str(image_path.absolute()))
            _, recon_dict = supplier.forward(image)
            # cv2.imshow("fr", fr)
            # cv2.waitKey(0)
            # self.frame_collection.append(fr)
            self.recon_collection.append(recon_dict)

            for part in recon_dict.keys():
                self.unique_parts.add(part)

                diff_img = cv2.subtract(np.array(recon_dict[part]["recon"]), np.array(recon_dict[part]["transformed_part"]))
                diff = np.argwhere(diff_img > 0).shape[0]
                self.data.append([part, diff])
                self.target.append(self.data_type)

        for index, element in enumerate(list(sorted(self.unique_parts))):
            self.mapper[element] = index

        for y in range(len(self.data)):
            self.data[y][0] = self.mapper[self.data[y][0]]

        self.data = np.array(self.data, dtype=np.float32)
        self.target = np.array(self.target, dtype=np.float32)

        assert (self.data.shape[0] == self.target.shape[0])

        print(f"Data shape {self.data.shape}")

    def get_map(self):
        return self.mapper

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index], self.get_map()

        # for recon in range(len(recon_dict.values())):
        #     recon_counter += 1
        #     self.mapper[recon_counter] = frame_counter
        # frame_counter += 1

        # self.clear_recon = []
        # for recon in self.recon_collection:
        #     for item in list(recon.values()):
        #         print(type(item))
        #         if self.transform:
        #             item = self.transform(item)
        #         self.clear_recon.append(item)

    # def __len__(self):
    #     return len(self.clear_recon)
    #
    # def __getitem__(self, index):
    #     return_data_type = self.data_type
    #     # if self.transform:
    #     #     return_data_type = self.transform(self.data_type)
    #
    #     return_data_type = self.caster(return_data_type)
    #     return self.clear_recon[index], return_data_type


class Dis(nn.Module):
    def __init__(self, in_f):
        super(Dis, self).__init__()
        self.in_f = in_f
        self.layers = nn.Sequential(
            nn.Linear(self.in_f, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_dis(model: Dis, train_data_loader, loss_function, lr, epochs=40):
    model = model.to(DEVICE)

    return_mapper = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    train_losses = []
    for epoch in range(epochs):
        print('Epoch: ', epoch)

        total_loss = 0.0

        print(len(train_data_loader))
        for batch, labels, mapper in train_data_loader:
            print(f"Batch type: {type(batch)}")
            print(f"Labels type: {type(labels)}")

            return_mapper = mapper

            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            # inputs, labels = batch
            # flatten_batch = torch.flatten(batch, start_dim=1)
            # print(f"flattt: {flatten_batch.size}")
            # assert (flatten_batch.(1) == model.in_f)
            outputs = model.forward(batch)
            print(f"output dtype: {outputs.dtype}")
            print(f"labels dtype: {labels.dtype}")

            temp = v2.ToDtype(dtype=torch.float, scale=False)
            labels = temp(labels)

            m = nn.Sigmoid()

            labels = labels.unsqueeze(1)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss)

    torch.save(
        {
            "model": model.state_dict(),
            "map": return_mapper
        },
        '../model_zoo/Discr/dis_model.pth')
    notify("Discr model finish", scenario='incomingCall')
    return model, train_losses


def test_dis(model, test_data_loader, loss_function):
    model.eval()

    losses = []

    for batch, labels, _ in test_data_loader:
        labels = labels.to(DEVICE)
        batch = batch.to(DEVICE)

        labels = labels.unsqueeze(1)

        outputs = model(batch)

        loss = loss_function(outputs, labels)
        losses.append(loss.item())

    plt.plot(losses)
    plt.title("Loss Function on test")
    plt.show()


img_transformer = v2.Compose([
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=True)
])


def split_len(dataset: DisDataset, test_proportion: float):
    test_len = int(test_proportion * len(dataset))
    train_len = len(dataset) - test_len
    return train_len, test_len


# if __name__ == "__main__":
#     import random
#     from PIL import Image
#     from video import VideoFrameExtract
#
#     demo = Demo()
#
#     video_reader = VideoFrameExtract()
#     video_reader.read("C:/Users/dabro/PycharmProjects/scientificProject/utildata/videos/Normal-001/000001.mp4")
#     frames, _ = video_reader.select_frames(10)
#
#     original, reconstructed = demo.forward(frames[0])
#
#     random.seed(42)
#     label = random.choice(list(reconstructed.keys()))
#
#     img = Image.fromarray(original)
#
#     print(reconstructed[label]["recon"].size)
#
#     color_coverted = reconstructed[label]["cutoff"]
#     cv2.imshow("lol", color_coverted)
#     cv2.waitKey(0)
#
#     color_coverted = color_coverted.astype(np.uint8)
#     a = Image.fromarray(color_coverted)
#     a.show()


if __name__ == "__main__":
    demo = Demo(rbm_model_path="../model_zoo/RBM/RBM_cuda_12_28_2023_14_41_54_uuid_f509f783.pth",
                unet_model_path="../model_zoo/UNet/unet_colab_t4_kxB53MMd.h5")

    path = "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/JPEGImages"
    broken_path = "C:/Users/dabro/PycharmProjects/scientificProject/data/damaged/data1a/training/00-damage"

    # data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model = Dis(2)

    temp_good = DisDataset(path, demo, 1, transform=img_transformer)
    temp_bad = DisDataset(broken_path, demo, 0, transform=img_transformer)

    test_prop = 0.2

    good, good_test = random_split(temp_good, split_len(temp_good, test_prop))
    bad, bad_test = random_split(temp_bad, split_len(temp_bad, test_prop))
    #
    con = ConcatDataset([good, bad])
    con_test = ConcatDataset([good_test, bad_test])

    # # train_data_loader = DataLoader(con, batch_size=1, shuffle=True)
    # # test_data_loader = DataLoader(con_test, batch_size=1, shuffle=True)
    #
    train_data_loader = DataLoader(con, batch_size=10, shuffle=True)
    test_data_loader = DataLoader(con_test, batch_size=1, shuffle=True)

    loss_f = torch.nn.BCELoss()

    post_train_model, train_losses = train_dis(model, train_data_loader, loss_f, 0.0001)

    plt.clf()
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.show()

    test_dis(post_train_model, test_data_loader, loss_f)
