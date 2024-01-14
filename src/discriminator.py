from torch.utils.data import Dataset
import os
from pathlib import Path

import cv2
from torch.utils.data import Dataset

from demo import DemoTransform
from models.RBM.base import RBM
from models.RBM.door_data import my_transforms
from src.models.blocks import SegmentationModel, ReconstructionModel

os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm


# https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/2
class PartsDataset(Dataset):
    def __init__(self, img_dir, parts_state_label: str, transform=None):
        if Path(img_dir).is_dir():
            self.img_dir = Path(img_dir)
        else:
            raise AttributeError("Directory not found")
        self.parts_state_label = parts_state_label
        self.transform = transform
        self._available_files = list(self.img_dir.glob('*.jpg'))

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

        for image_path in self._available_files:
            image_path_str = str(image_path.absolute())
            frame = cv2.imread(image_path_str)
            _, mask, legends = self.seg_block.generate_masks(frame)
            data = DemoTransform(frame, mask, legends, transform=my_transforms)

            self.reconstructed_parts = dict()

            for part_no in range(len(data)):
                _, temp = self.recon_block.reconstruct(*data[part_no])
                self.reconstructed_parts.update(temp)

    def __len__(self):
        return len(list(self.reconstructed_parts.values()))

    def __getitem__(self, index):
        return list(self.reconstructed_parts.values())[index]
