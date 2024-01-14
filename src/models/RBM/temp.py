from torch.utils.data import DataLoader

from door_data import CarDataset, train_transforms

from base import RBM, InspectModel, test
from torch.utils.data import DataLoader

from base import RBM, InspectModel, test
from door_data import CarDataset, train_transforms

inspc = InspectModel(RBM, ["epochs_number", "train series UUID", "parts"],
                         "C:/Users/dabro/OneDrive/Pulpit/test_trained/RBM_cuda_12_10_2023_16_07_27_uuid_4aff48c0.pth"
                     )

model = inspc.get_model()

test_data = CarDataset(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/annotations.json",
        parts=["front_left_door", 'front_right_door', 'hood', "front_bumper", 'back_bumper', 'tailgate'],
        transform=train_transforms
    )

test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

test(model, test_loader, f"on_test_k_hope_works.jpg", f"loss_on_test_k_hope_works.jpg")
