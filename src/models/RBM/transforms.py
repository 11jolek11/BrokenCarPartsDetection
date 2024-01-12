from typing import Tuple
import cv2
import numpy as np
import torch


class Canny(torch.nn.Module):
    def __init__(self, threshold: Tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def forward(self, sample):
        edges = cv2.Canny(image=sample, threshold1=self.threshold[0], threshold2=self.threshold[1])
        return edges

# TODO(11jolek11): Zrobić tak jak u Łukaszewskiego filtrować wartości RGB i podbijać gradient na krawędziach
class GaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size: Tuple[int, int], sigma_x=0, sigma_y=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def forward(self, sample):
        return cv2.GaussianBlur(sample, self.kernel_size, sigmaX=self.sigma_x, sigmaY=self.sigma_y)


class Binarize(torch.nn.Module):
    def __init__(self, threshold=100, value_on_positive: float = 1.0, value_on_negative: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.value_on_positive = value_on_positive
        self.value_on_negative = value_on_negative

    def forward(self, sample, *args, **kwargs):
        return np.where(sample > self.threshold, self.value_on_positive, self.value_on_negative)


class FindBoundingBoxAndCrop(torch.nn.Module):
    def __init__(self, max_value=255, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MIN_CONTOUR_AREA = 300
        self.max_value = max_value

    def forward(self, sample, *args, **kwargs):
        cropped_image = sample
        img_thresh = cv2.adaptiveThreshold(sample, self.max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > self.MIN_CONTOUR_AREA:
                [X, Y, W, H] = cv2.boundingRect(contour)
                cropped_image = sample[Y:Y + H, X:X + W]
        return cropped_image


class Resize(torch.nn.Module):
    def __init__(self, new_size: Tuple[int, int], interpolation: int = cv2.INTER_AREA, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_size = new_size
        self.interpolation = interpolation

    def forward(self, sample):
        new_sample = cv2.resize(sample, self.new_size, interpolation=self.interpolation)
        return new_sample


class BilateralFilter(torch.nn.Module):
    def __init__(self, bilateral_filter, sigma_color, sigma_space):
        super().__init__()
        self.bilateral_filter = bilateral_filter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def forward(self, sample):
        return cv2.bilateralFilter(sample, self.bilateral_filter, self.sigma_color, self.sigma_space)


class Erode(torch.nn.Module):
    def __init__(self, kernel: Tuple[int, int]):
        # TODO(11jolek11): think about diff parameters:
        # https://pythonexamples.org/python-opencv-image-erosion/
        super().__init__()
        self.kernel = kernel

    def forward(self, sample):
        return cv2.erode(sample, self.kernel)


class ColorToHSV(torch.nn.Module):
    def __init__(self, target_color_space=cv2.COLOR_BGR2HSV):
        super().__init__()
        self.target_color_space = target_color_space

    def forward(self, sample):
        return cv2.cvtColor(sample, self.target_color_space)


class RemoveInnerContours(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        edges = cv2.Canny(sample, 50, 150, apertureSize=3)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Create an empty mask
        mask = np.zeros_like(edges)

        # Iterate through contours
        for i in range(len(contours)):
            # Check if the contour has no parent (outer contour)
            if hierarchy[0][i][3] == -1:
                # Draw the contour on the mask
                cv2.drawContours(mask, contours, i, 255, thickness=cv2.FILLED)

        # Bitwise AND operation to keep only the outer contours
        result = cv2.bitwise_and(edges, mask)
        return result

