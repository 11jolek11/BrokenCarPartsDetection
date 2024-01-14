import os
from pathlib import Path

import cv2


class Scaler:

    def __init__(self, end_size: tuple[int], interpolator=cv2.INTER_LANCZOS4, read_mode=cv2.IMREAD_UNCHANGED):
        # For image up-scaling try interpolator=cv2.INTER_LINEAR
        self.end_size = end_size
        self.interpolator = interpolator
        self.read_mode = read_mode

    def scale(self, path, save_path, show=False):
        # TODO(11jolek11): path change to object Path
        img = cv2.imread(path, self.read_mode)
        resized = cv2.resize(img, self.end_size, interpolation=self.interpolator)
        if show:
            cv2.imshow("Downscaled image", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imwrite("C:/Users/dabro/Videos/Movies/mickymouse/resized.jpg", resized) <-- not working
            # return resized
        # return resized
        cv2.imwrite(save_path, resized)


class FileManipulator:
    def __init__(self, root, save_path):
        self.root = root
        self.save_path = Path(save_path).absolute()
        # TODO(11jolek11): reimplement using generators here
        self.discovered_files = []

    def discover(self):
        os.chdir(self.root)
        current_path = Path("")
        self.discovered_files = list(current_path.glob("*.jpg"))

    def apply(self):
        sc = Scaler((128, 128))
        for str_path in self.discovered_files:
            sc.scale(str(str_path), str(str_path))


if __name__ == "__main__":
    f = FileManipulator("C:/Users/dabro/PycharmProjects/scientificProject/utildata/from_broken/", "C:/Users/dabro/PycharmProjects/scientificProject/data/from_broken/")
    f.discover()
    f.apply()
    # sc = Scaler((128, 128))
    # sc.scale("../utildata/broken/archive/img/1.jpg", show=True)
