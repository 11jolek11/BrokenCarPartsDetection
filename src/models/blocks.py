from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


class SegmentationModel:
    def __init__(self, class_names, model_class, model_path: Path, *args, **kwargs) -> None:
        self.model_path = model_path
        self.class_names = class_names
        self._model_lib = "torch" if model_path.suffix == ".pth" else "tensorflow"
        self.model = None
        if self._model_lib == "tensorflow":
            model_class = model_class(*args, **kwargs)
            model_class.load_weights(self.model_path.absolute())
            self.model = model_class

        if self._model_lib == "torch":
            raise NotImplementedError("Torch models handling is not supported yet")

    def preprocess_image(self, img):
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        ww = 512
        hh = 512
        img.thumbnail((hh, ww))
        i = np.array(img)
        ht, wd, cc = i.shape

        # create new image of desired size and color (blue) for padding
        color = (0, 0, 0)
        result = np.full((hh, ww, cc), color, dtype=np.uint8)

        # copy img image into center of result image
        result[:ht, :wd] = img
        return result, ht, wd

    def get_class_names_to_slice(self, class_names, tags):

        n_classes = len(class_names)
        legend = dict()

        class_names_colors = enumerate(class_names[:n_classes])

        for (i, class_name) in class_names_colors:
            if i in tags:
                legend[class_name] = i

        return legend

    def generate_masks(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        tags = []
        print(frame.size)
        img_scaled_arr = self.preprocess_image(frame)
        print(frame.size)
        image = np.expand_dims(img_scaled_arr[0], axis=0)
        # FIXME(11jolek11): pr_mask bad z dimensions
        pr_mask = self.model.predict(image).squeeze()
        pr_mask_int = np.zeros((pr_mask.shape[0], pr_mask.shape[1]))
        kernel = np.ones((5, 5), 'uint8')
        for i in range(1, 19):
            array_one = np.round(pr_mask[:, :, i])
            op = cv2.morphologyEx(array_one, cv2.MORPH_OPEN, kernel)
            if sum(sum(op == 1)) > 100:
                tags.append(i)
                pr_mask_int[op == 1] = i

        # img_segmented = np.array(Image.fromarray(pr_mask_int[:img_scaled_arr[1], :img_scaled_arr[2]]).resize(frame.size))

        return frame, pr_mask, self.get_class_names_to_slice(self.class_names, tags)


class ReconstructionModel:
    def __init__(self, model_class, model_path: Path, *args, **kwargs) -> None:
        self.model_path = model_path
        self._model_lib = "torch" if model_path.suffix == ".pth" else "tensorflow"
        self.model = None
        if self._model_lib == "torch":
            model_class = model_class(*args, **kwargs)
            model_class.load_state_dict(torch.load(model_path.absolute())['model_state_dict'])
            self.model = model_class
            self.model.eval()

        if self._model_lib == "tensorflow":
            raise NotImplementedError("Tensorflow models handling is not supported yet")

    def reconstruct(self, frame: np.ndarray, label: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        reconstructed_frame = self.model.reconstruct(frame)
        temp = dict()
        temp[label] = reconstructed_frame
        return frame, temp
