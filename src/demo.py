import numpy as np

from models.blocks import ReconstructionModel, SegmentationModel
from models.RBM.base import RBM
from models.RBM.door_data import my_transforms
from pathlib import Path
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from torch.utils.data import Dataset


class DemoTransform(Dataset):
    def __init__(self, frame, masks, legend: dict[str, np.ndarray], transform=None):
        self.frame = frame
        self.masks = masks
        self.legend = legend
        self.masks_selection = tuple(self.legend.keys())
        self.transform = transform

    def __len__(self):
        return len(self.masks_selection)

    def __getitem__(self, index):
        mask = self.masks[:, :, self.legend[self.masks_selection[index]]]
        cut_first = self.frame[:, :, 0] * mask
        cut_second = self.frame[:, :, 1] * mask
        cut_third = self.frame[:, :, 2] * mask

        img = np.dstack((cut_first, cut_second, cut_third))

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

    def forward(self, frame: np.ndarray):
        _, mask, legends = self.seg_block.generate_masks(frame)
        data = DemoTransform(frame, mask, legends, transform=my_transforms)

        reconstructed_parts = dict()

        for part_no in range(len(data)):
            _, temp = self.recon_block.reconstruct(*data[part_no])
            reconstructed_parts.update(temp)

        return frame, reconstructed_parts



if __name__ == "__main__":
    demo = Demo()
