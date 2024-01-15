from pathlib import Path

import torch

from base import RBM


class InspectModel:
    def __init__(self, model, keywords: list[str] | tuple[str], path_to_model: str | Path = None) -> None:
        self.model = model
        self.keywords = keywords
        self.path_to_model = path_to_model

    def set_path_to_model(self, path_to_model: str | Path) -> None:
        self.path_to_model = path_to_model

    def get_model(self):
        state_dicta = torch.load(self.path_to_model)['model_state_dict']
        # print(state_dicta.keys())
        # odict_keys(['weight', 'bias_v', 'bias_h', 'bias v', 'bias h', 'weights'])
        # print(state_dicta['bias_v'].shape[1])
        # print(state_dicta['bias_h'].shape[1])
        model = self.model(state_dicta['bias_v'].shape[1], state_dicta['bias_h'].shape[1])
        # state_dicta = torch.load(self.path_to_model)
        # print(state_dicta)
        temp = model.load_state_dict(state_dicta)

        return temp


if __name__ == '__main__':
    inspc = InspectModel(RBM,
                         ["epochs_number", "train series UUID", "parts"],
                         "C:/Users/dabro/PycharmProjects/scientificProject/models/RBM_cuda_12_04_2023_17_58_47.pth")
    inspc.get_model()
