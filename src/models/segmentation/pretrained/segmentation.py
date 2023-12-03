import torch
from torchvision import transforms
from torch import nn
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import panopticapi
from panopticapi.utils import rgb2id
import io
from copy import deepcopy
from detectron2.data.datasets import register_coco_instances,register_coco_panoptic, load_coco_json
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from pathlib import Path
import pprint


image_normalize = transforms.Compose([
    transforms.Resize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DETR:
    def __init__(self, backbone: str, dataset_name: str, annotations_file: Path, image_dir: Path):
        self.model, self.postprocessor = torch.hub.load('facebookresearch/detr', backbone, pretrained=True, return_postprocessor=True)
        self.model.eval()
        self._dataset_name = dataset_name
        self._annotations_file = annotations_file
        self._image_dir = image_dir

        register_coco_instances(dataset_name, {}, annotations_file, image_dir)
        dataset = load_coco_json(annotations_file, image_dir, dataset_name=self._dataset_name)
        # register_coco_panoptic(dataset_name, {}, annotations_file, image_dir)
    def generate_mask(self, input_image: Image):
        image = image_normalize(input_image).unsqueeze(0)
        out = self.model(image)
        result = self.postprocessor(out, torch.as_tensor(image.shape[-2:]).unsqueeze(0))[0]
        panoptic_seg = Image.open(io.BytesIO(result['png_string']))
        final_w, final_h = panoptic_seg.size
        panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
        panoptic_seg = torch.from_numpy(panoptic_seg)

        segments_info = deepcopy(result["segments_info"])

        # FIXME(11jolek11): Assign id's correctly
        meta = MetadataCatalog.get(self._dataset_name)

        plt.imshow(panoptic_seg)
        plt.axis('off')
        plt.show()

        # print(segments_info)
        # pp = pprint.PrettyPrinter()
        # pp.pprint(segments_info)
        print(meta.thing_dataset_id_to_contiguous_id)

        # for i in range(len(segments_info)):
        #     c = segments_info[i]["category_id"]
        #     segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

        # # Finally we visualize the prediction
        # v = Visualizer(numpy.array(input_image.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
        # v._default_font_size = 20
        # v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
        # img = Image.fromarray(v.get_image(), 'RGB')
        # img.show()

if __name__ == "__main__":
    p = DETR("detr_resnet101_panoptic",
             "Car",
             "C:/Users/dabro/PycharmProjects/scientificProject/notebooks/output.json",
             # "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
             "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/JPEGImages")
    with Image.open("C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/JPEGImages/car4.jpg") as im:
        p.generate_mask(im)
