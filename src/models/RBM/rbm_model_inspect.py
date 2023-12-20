import torch
from pathlib import Path


class InspectModel:
    def __init__(self, model, keywords: list[str] | tuple[str], path_to_model: str | Path = None) -> None:
        self.model = model
        self.keywords = keywords
        self.path_to_model = path_to_model

    def set_path_to_model(self, path_to_model: str | Path) -> None:
        self.path_to_model = path_to_model

    def get_model(self):
        state_dicta = torch.load(self.path_to_model)["model_state_dict"]
        # state_dicta = torch.load(self.path_to_model)
        # print(state_dicta)
        # model = self.model.load_state_dict(state_dicta)


if __name__ == '__main__':
    inspc = InspectModel(RBM,
                         ["epochs_number", "train series UUID", "parts"],
                         "C:/Users/dabro/OneDrive/Pulpit/test_trained/RBM_cuda_12_10_2023_16_07_27_uuid_4aff48c0.pth")
    inspc.get_model()
