import torch
from pathlib import Path


class InspectModel:
    def __init__(self, model: torch.nn.Module, keywords: list[str] | tuple[str], path_to_model: str | Path = None) -> None:
        self.model = model
        self.keywords = keywords
        self.path_to_model = path_to_model

    def set_path_to_model(self, path_to_model: str | Path) -> None:
        self.path_to_model = path_to_model

    def get_model(self):
        model = self.model.load_state_dict(torch.load(self.path_to_model))


# if __name__ == '__main__':
    # inspc = InspectModel(None,
    #                      ["epochs_number", "train series UUID", "parts"],
    #                      "C:\Users\dabro\OneDrive\Pulpit\test_trained\RBM_cuda_12_10_2023_16_07_27_uuid_4aff48c0.pth")
    # inspc.info()
