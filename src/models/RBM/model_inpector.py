from torch.utils.data import DataLoader
from base import RBM
from door_data import DoorsDataset3, door_transforms
import torchvision.transforms as T
import torch
from src.settings import DEVICE


model = RBM(128*128, 128*128+600, k=33)
model.load_state_dict(torch.load("../../../models/RBM_cuda_12_04_2023_17_58_47.pth")['model_state_dict'])
model = model.to(DEVICE)
model.eval()

test_data = DoorsDataset3(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/annotations.json",
        transform=door_transforms
    )

test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

test = test_data[0]
make_visible = T.ToPILImage()
make_visible(test).show("test_image")
test = test.reshape((-1, model.number_of_visible))
test = test.to(DEVICE)
recon = model.forward(test, k=1)
recon = recon.reshape((1, 128, 128))
# test_tr = test.reshape((1, 128, 128))
recon = recon.detach()
make_visible(recon).show("recon")
