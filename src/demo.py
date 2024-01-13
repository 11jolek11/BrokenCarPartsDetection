import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2
from settings import DEVICE


from models.blocks import ReconstructionModel, SegmentationModel
from models.RBM.base import RBM
from models.RBM.door_data import my_transforms, my_transforms_only_image
from pathlib import Path
import os
import torch.nn as nn
from torchvision.transforms import v2


os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import torch


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

        img = (img * 255).astype(np.uint8)

        if self.transform:
            img = self.transform(img)

        return img, self.masks_selection[index]


class Demo:
    def __init__(self):
        self.recon_model_path = Path(
            "C:/Users/dabro/PycharmProjects/scientificProject/models/RBM_cuda_12_28_2023_14_41_54_uuid_f509f783.pth")
        self.segmentation_model_path = Path("C:/Users/dabro/Downloads/best_model (1).h5")

        # ['_background_', 'back_bumper', 'back_glass', 'back_left_door', 'back_left_light', 'back_right_door',
        # 'back_right_light', 'front_bumper', 'front_glass', 'front_left_door', 'front_left_light', 'front_right_door',
        # 'front_right_light', 'hood', 'left_mirror', 'right_mirror', 'tailgate', 'trunk', 'wheel']
        self.class_names = ['_background_', 'back_bumper', 'back_glass', 'back_left_door', 'back_left_light',
                            'back_right_door',
                            'back_right_light', 'front_bumper', 'front_glass', 'front_left_door', 'front_left_light',
                            'front_right_door',
                            'front_right_light', 'hood', 'left_mirror', 'right_mirror', 'tailgate', 'trunk', 'wheel']

        assert (len(self.class_names) == 19)

        self.seg_model = sm.Unet
        self.recon_model = RBM
        # (128 * 128, 128*128, k=3)

        self.seg_block = SegmentationModel(self.class_names, self.seg_model, self.segmentation_model_path,
                                           'resnet18', classes=19, activation='softmax')

        self.recon_block = ReconstructionModel(RBM, self.recon_model_path, 128 * 128, 128 * 128, k=3)

    def forward(self, frame: np.ndarray, to_image: bool = True):
        _, mask, legends = self.seg_block.generate_masks(frame)
        data = DemoTransform(frame, mask, legends, transform=my_transforms)

        reconstructed_parts = dict()

        for part_no in range(len(data)):
            data_temp, mask_temp = data[part_no]

            _, temp = self.recon_block.reconstruct(*data[part_no])
            reconstructed_parts.update(temp)

        return frame, reconstructed_parts


class DisDataset(Dataset):
    def __init__(self, data_path: str | Path, supplier: Demo, data_type, transform=None):
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

        frame_counter = 0
        recon_counter = 0

        for image_path in self._available_files:
            image = cv2.imread(str(image_path.absolute()))
            fr, recon_dict = supplier.forward(image)
            self.frame_collection.append(fr)
            self.recon_collection.append(recon_dict)

            for recon in range(len(recon_dict.values())):
                recon_counter += 1
                self.mapper[recon_counter] = frame_counter
            frame_counter += 1

        self.clear_recon = []
        for recon in self.recon_collection:
            for item in list(recon.values()):
                if self.transform:
                    item = self.transform(item)

                    # FIXME(11jolek11): self.data_type is int not torch.Tensor with dtype.float32
                self.clear_recon.append(item)

    def __len__(self):
        return len(self.clear_recon)

    def __getitem__(self, index):
        return_data_type = self.data_type
        # if self.transform:
        #     return_data_type = self.transform(self.data_type)

        return_data_type = self.caster(return_data_type)
        return self.clear_recon[index], return_data_type


class Dis(nn.Module):
    def __init__(self, in_f):
        super(Dis, self).__init__()
        self.in_f = in_f
        self.layers = nn.Sequential(
            nn.Linear(self.in_f, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_dis(model: Dis, train_data_loader, loss_function, lr, epochs=40):
    model = model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()

    train_losses = []
    for epoch in range(epochs):
        print('Epoch: ', epoch)

        total_loss = 0.0

        print(len(train_data_loader))
        for batch, labels in train_data_loader:
            print(f"Batch type: {type(batch)}")
            print(f"Labels type: {type(labels)}")
            labels = labels.reshape(-1, 1)
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            # inputs, labels = batch
            flatten_batch = torch.flatten(batch,start_dim=1)
            print(f"flattt: {flatten_batch.size}")
            # assert (flatten_batch.(1) == model.in_f)
            outputs = model(flatten_batch)
            print(f"output dtype: {outputs.dtype}")
            print(f"labels dtype: {labels.dtype}")

            temp = v2.ToDtype(dtype=torch.float, scale=False)
            labels = temp(labels)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss)

    torch.save(model.state_dict(), './dis_model.pth')
    return model, train_losses


def test_dis(model, test_data_loader, loss_function):
    model.eval()

    losses = []

    for _, data in enumerate(test_data_loader):
        inputs, labels = data
        outputs = model.forward(inputs)
        loss = loss_function(outputs, labels)
        losses.append(loss)

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


if __name__ == "__main__":
    import random
    from PIL import Image
    from video import VideoFrameExtract

    demo = Demo()

    video_reader = VideoFrameExtract()
    video_reader.read("C:/Users/dabro/PycharmProjects/scientificProject/data/videos/Normal-001/000001.mp4")
    frames, _ = video_reader.select_frames(10)
    fr, recon_dict = demo.forward(frames[0])

    test_image = recon_dict[random.choice(list(recon_dict.keys()))]
    print(len(recon_dict.values()))
    test_image.show()

    original, reconstructed = demo.forward(frames[0])

    label = random.choice(list(reconstructed.keys()))

    img = Image.fromarray(original)
    img.show(f"Image of original {label}")

    reconstructed[label].show(f"Image of reconstructed {label}")
    print(reconstructed[label].size)

    print(f"Conf type: {type(reconstructed[label])}")
    reconstructed[label].show(f"Image of reconstructed {label}")
