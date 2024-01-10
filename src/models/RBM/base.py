import os
import shutil
from datetime import datetime
import uuid
from tqdm import tqdm
from alive_progress import alive_bar
from win11toast import notify

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pathlib import Path

from .door_data import DoorsDataset2, DoorsDataset3, door_transforms, door_transforms2, CarDataset, train_transforms, my_transforms
from src.settings import DEVICE

# folder = "../../../logs/runs/"
# for filename in os.listdir(folder):
#     file_path = os.path.join(folder, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))
# summary_writer = SummaryWriter("../../../logs/runs/", flush_secs=30)


class RBM(nn.Module):
    def __init__(self, number_of_visible, number_of_hidden, k=3, *args, **kwargs):
        super(RBM, self).__init__()
        self.number_of_visible = number_of_visible
        self.number_of_hidden = number_of_hidden
        self._k = k
        self.weight = nn.Parameter(torch.rand(number_of_hidden, number_of_visible)*1e-9)
        self.bias_v = nn.Parameter(torch.rand(1, number_of_visible)*1e-4)  # Bias for visible lay.
        self.bias_h = nn.Parameter(torch.rand(1, number_of_hidden)*1e-4)  # Bias for hidden lay.
        self.register_parameter("bias v", self.bias_v)
        self.register_parameter("bias h", self.bias_h)
        self.register_parameter("weights", self.weight)

    def get_properties(self):
        hiper_params = [self.number_of_hidden, self.number_of_visible, self._k]
        hiper_params_labels = ["hidden units", "visible units", "k"]
        return dict(zip(hiper_params_labels, hiper_params))

    def sample_h_from_v(self, v0):
        # print(f"V0 type: {v0.dtype} self.weight type: {self.weight.dtype} self.bias_h type: {self.bias_h.dtype}")
        phv = torch.sigmoid(nn.functional.linear(v0, self.weight, self.bias_h))
        # h = nn.functional.relu(phv)
        h = torch.bernoulli(phv)
        return h, phv

    def sample_v_from_h(self, h0):
        # print(f"H0 type: {h0.dtype} self.weight type: {self.weight.dtype} self.bias_v type: {self.bias_v.dtype}")
        pvh = torch.sigmoid(nn.functional.linear(h0, self.weight.T, self.bias_v))
        # v = nn.functional.relu(pvh)
        v = torch.bernoulli(pvh)
        return v, pvh

    def forward(self, v, k=None):
        if not k:
            k = self._k
        h, phv = self.sample_h_from_v(v)
        # CD-k algo
        for i in range(k):
            v, pvh = self.sample_v_from_h(phv)
            h, phv = self.sample_h_from_v(v)
        v, pvh = self.sample_v_from_h(phv)
        return v

    # TODO(11jolek11): Use reconstruction loss as a loss?
    # loss aka "free energy"
    def loss(self, v):
        vt = torch.matmul(v, torch.t(self.bias_v))
        temp = torch.log(torch.add(torch.exp(nn.functional.linear(v, self.weight, self.bias_h)), 1))
        ht = torch.sum(temp, dim=1)
        return -vt - ht

    def reconstruct(self, image, size=128, to_image: bool = True):
        self.to(DEVICE)
        self.eval()
        image = image.reshape((-1, self.number_of_visible))
        image = image.to(DEVICE)

        v = self.forward(image, k=self._k).to(DEVICE)
        # TODO(11jolek11): Add better size handling
        v = v.reshape((1, size, size))

        if to_image:
            transformer = T.ToPILImage()
            temp = transformer(v)
            return temp

        return v


def creat_folder(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")


def train(model, data_loader, lr, epochs_number: int, parts: [str], optimizer, *args, **kwargs):
    train_uuid = str(uuid.uuid4())[:8]
    print(f"Train series UUID {train_uuid}")

    checkpoint_path = f"../../../checkpoint/{train_uuid}/"
    summary_path = f"../../../logs/runs/{train_uuid}/"

    creat_folder(checkpoint_path)
    creat_folder(summary_path)

    summary_writer = SummaryWriter(summary_path, flush_secs=30)

    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    all_loss_by_epoch = []
    for epoch in range(epochs_number):
        total_loss_by_epoch = 0

        # for batch in data_loader:
        with tqdm(data_loader, unit="batch") as tepoch:
            for batch in tepoch:
                # NOTE: technicznie to nie batch tylko pojedyncze zdjęcie bo na tym RBM się może uczyć
                # poza tym cały dataset mógłby nie zmieścić się w pamięci
        # with alive_bar(len(data_loader), title=f"Epoch [{epoch}/{epochs_number}] ") as bar:
        #     for batch in data_loader:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs_number} ")
                batch = batch.reshape((-1, model.number_of_visible))
                batch = batch.to(DEVICE)

                v = model.forward(batch).to(DEVICE)

                loss = torch.mean(model.loss(batch)) - torch.mean(model.loss(v))

                # print(f"Loss: {loss} Epoch [{epoch + 1}/{epochs_number}]")
                total_loss_by_epoch += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # tepoch.set_postfix(loss=loss.item())
                tepoch.set_postfix_str(f" loss = {loss.item()} \n", refresh=True)
                # bar()
            summary_writer.flush()

                # summary_writer.add_graph(model, batch)

        summary_writer.add_scalar("Loss", total_loss_by_epoch, epoch)
        print('Epoch [{}/{}] --> loss: {:.4f}'.format(epoch, epochs_number, total_loss_by_epoch))
        all_loss_by_epoch.append(total_loss_by_epoch)
        now = datetime.now()
        torch.save({
            "epoch": epoch,
            "train series UUID": train_uuid,
            "parts": parts,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss_by_epoch,
        }, f"{checkpoint_path}checkpoint_RBM_{DEVICE.type}_{now.strftime('%m_%d_%Y_%H_%M_%S')}_epoch{epoch}.pth")

    torch.save({
        "epochs_number": epochs_number,
        "train series UUID": train_uuid,
        "parts": parts,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"../../../models/RBM_{DEVICE.type}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}_uuid_{train_uuid}.pth")

    hpams = model.get_properties()
    hpams["epochs_number"] = epochs_number
    summary_writer.add_hparams(hpams, {"all_loss_by_epoch": np.array(min(all_loss_by_epoch))}, run_name=train_uuid)
    print(f"Train series UUID {train_uuid}")
    # summary_writer
    summary_writer.close()

    plt.plot(all_loss_by_epoch)
    plt.savefig(f"all_loss_by_epoch_{train_uuid}")
    notify("RBM model finish", scenario='incomingCall')


def test(model, data_loader, file_name, loss_file_name, parts, size=128, k=None):
    model = model.eval()
    model = model.to(DEVICE)

    file_name = file_name + str(parts) + ".jpg"

    if not k:
        k = model._k

    list_img = []
    losses = []

    for batch in data_loader:
        original_batch = torch.clone(batch).reshape((1, size, size))
        original_batch = original_batch.to(DEVICE)
        batch = batch.reshape((-1, model.number_of_visible))
        batch = batch.to(DEVICE)

        v = model.forward(batch, k=k).to(DEVICE)
        # loss = torch.mean(model.loss(batch)) - torch.mean(model.loss(v))
        # losses.append(loss.resize((1, -1)).item())
        v = v.reshape((1, size, size))

        list_img.append(original_batch)
        list_img.append(v)

    # plt.plot(losses)
    # plt.savefig(loss_file_name)
    grid = make_grid(list_img, nrow=2)
    # grid.resize((len(list_img)/2, 1, 128, 128))
    # grid = make_grid(list_img)
    # summary_writer.add_images("Compare", grid)
    img = T.ToPILImage()(grid)
    img.save(file_name)
    print(f"Image saved to {file_name}")

def test3(model, data_loader, file_name, loss_file_name, parts, size=128):
    model = model.eval()
    model = model.to(DEVICE)

    file_name = file_name + str(parts)

    list_img = []
    losses = []

    for batch in data_loader:
        original_batch = torch.clone(batch).reshape((1, size, size))
        original_batch = original_batch.to(DEVICE)
        batch = batch.reshape((-1, model.number_of_visible))
        batch = batch.to(DEVICE)

        v = model.forward(batch).to(DEVICE)
        # v = batch*model.weight

        # loss = torch.mean(model.loss(batch)) - torch.mean(model.loss(v))
        # losses.append(loss.resize((1, -1)).item())
        v = v.reshape((1, size, size))

        list_img.append(original_batch)
        list_img.append(v)

    # plt.plot(losses)
    # plt.savefig(loss_file_name)
    grid = make_grid(list_img, nrow=2)
    # grid.resize((len(list_img)/2, 1, 128, 128))
    # grid = make_grid(list_img)
    # summary_writer.add_images("Compare", grid)
    img = T.ToPILImage()(grid)
    img.save(file_name)


if __name__ == "__main__":

    # back_right_door - - 29
    # back_right_light - - 20
    # front_bumper - - 74
    # front_glass - - 74
    # front_left_door - - 35
    # front_left_light - - 62
    # front_right_door - - 39
    # front_right_light - - 67
    # hood - - 74
    # left_mirror - - 55
    # right_mirror - - 60
    # tailgate - - 7
    # trunk - - 18
    # wheel - - 76

    # parts = ["front_right_door", "hood"],
    input_shape = (128, 128)
    datas = CarDataset(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
        parts=["hood", "front_glass", "back_glass", "left_mirror", "right_mirror",
               "front_left_door", "front_right_door", "back_left_door", "back_left_light",
               "back_right_door", "back_right_light"],
        transform=my_transforms
    )
    #
    print(f'datas size {len(datas)}')

    if len(datas) < 10:
        raise ValueError('Not enough data')

    train_loader = DataLoader(dataset=datas, batch_size=1, shuffle=True)

    # make_visible = T.ToPILImage()
    # for number, batch in enumerate(datas):
    #     print(batch.shape)
    #     # batch = batch.resize((1, 128, 128))
    #     make_visible(batch).save(f"C:/Users/dabro/PycharmProjects/scientificProject/data/transformed/image_{number}.jpg")
    #
    #
    # import sys
    # sys.exit(0)

    #
    # # my_model = RBM(128*128, 128*128+600, k=33)
    # # train(my_model, train_loader, 0.001, 30, torch.optim.SGD)
    #
    # # my_model = RBM(128 * 128, 800, k=3)
    # # train(my_model, train_loader, 0.001, 19, torch.optim.SGD)
    #
    # # my_model = RBM(128 * 128, 128 * 128, k=3)
    # # train(my_model, train_loader, 0.001, 1, torch.optim.SGD)
    #
    # my_model = RBM(128 * 128, 400, k=3)
    # train(my_model, train_loader, 0.001, 80, datas.get_parts(), torch.optim.SGD)
    # # 6
    #
    # test_data = DoorsDataset3(
    #     "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/",
    #     "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/annotations.json",
    #     transform=door_transforms
    # )
    #
    # test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    #
    # test(my_model, test_loader, "on_test.jpg")
    #
    # test(my_model, train_loader, "on_train.jpg")
    #
    # datas_test = DoorsDataset3(
    #     "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
    #     "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
    #     transform=door_transforms
    # )
    #
    # test = datas_test[0]
    # make_visible = T.ToPILImage()
    # make_visible(test).save("testttth.jpg")
    # test = test.reshape((-1, my_model.number_of_visible))
    # test = test.to(DEVICE)
    # recon = my_model.forward(test)
    # recon = recon.reshape((1, 128, 128))
    # # test_tr = test.reshape((1, 128, 128))
    # recon = recon.detach()
    # make_visible(recon).save("recon.jpg")
    # # print(f"TEST IMAGE SHAPE {test.shape} -- {recon.shape}")
    # # grid = make_grid([test_tr, recon])
    # # img = T.ToPILImage()(grid)
    # # img.show()

    # datas = DoorsDataset3(
    #     "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
    #     "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
    #     transform=door_transforms
    # )
    # datas = DoorsDatasetFromFiles(
    #     "C:/Users/dabro/PycharmProjects/scientificProject/notebooks/CarPartsDatasetExperimentDir/exp2/output",
    #     transform=door_transforms
    # )
    # train_loader = DataLoader(dataset=datas, batch_size=1, shuffle=True)

    # my_model = RBM(128*128, 128*128+600, k=33)
    # train(my_model, train_loader, 0.001, 30, torch.optim.SGD)

    # my_model = RBM(128 * 128, 800, k=3)
    # train(my_model, train_loader, 0.001, 19, torch.optim.SGD)

    # my_model = RBM(128 * 128, 128 * 128, k=3)
    # train(my_model, train_loader, 0.001, 1, torch.optim.SGD)


    my_model = RBM(128 * 128, 128*128, k=3)

    # my_model = RBM(32 * 32, 512, k=4)
    # epochs number 11
    train(my_model, train_loader, 0.001, 25, datas.get_parts(), torch.optim.SGD)

    test_data = CarDataset(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/annotations.json",
        parts=["hood"],
        transform=my_transforms
    )
    print(f'test data size {len(test_data)}')
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

    test(my_model, test_loader, "on_test_new_", None, ["hood"])

    test_data = CarDataset(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/annotations.json",
        parts=["front_glass"],
        transform=my_transforms
    )

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

    test(my_model, test_loader, "on_test_new_", None, ["front_glass"])

    test_data = CarDataset(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/annotations.json",
        parts=["left_mirror"],
        transform=my_transforms
    )

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

    test(my_model, test_loader, "on_test_new_", None, ["left_mirror"])

    test_data = CarDataset(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/annotations.json",
        parts=["front_left_door"],
        transform=my_transforms
    )

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

    test(my_model, test_loader, "on_test_new_", None, ["front_left_door"])

    test_data = CarDataset(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/annotations.json",
        parts=["back_left_light"],
        transform=my_transforms
    )

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

    test(my_model, test_loader, "on_test_new_", None, ["back_left_light"])


    # back_left_light
    # make_visible = T.ToPILImage()
    # for number, batch in enumerate(test_data):
    #     print(batch.shape)
    #     # batch = batch.resize((1, 128, 128))
    #     make_visible(batch).save(
    #         f"C:/Users/dabro/PycharmProjects/scientificProject/data/transformed_test/image_{number}.jpg")
    #
    # import sys
    #
    # sys.exit(0)
































    # test_k = None

    # test(my_model, test_loader, f"on_test_k_my_tr_{test_k}.jpg", f"loss_on_test_k_my_tr_{test_k}.jpg", k=test_k)

    # test(my_model, train_loader, f"on_train_k_my_tr_{test_k}.jpg", f"loss_on_train_my_tr_k_{test_k}.jpg", k=test_k)

    # datas_test = DoorsDataset3(
    #     "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
    #     "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
    #     parts=["front_left_door", 'front_right_door', 'hood', "front_bumper", 'back_bumper', 'tailgate'],
    #     transform=train_transforms
    # )

    # test = datas_test[0]
    # make_visible = T.ToPILImage()
    # make_visible(test).save("testttth32_32__aug_k_1.jpg")
    # test = test.reshape((-1, my_model.number_of_visible))
    # test = test.to(DEVICE)
    # recon = my_model.forward(test)
    # recon = recon.reshape((1, *input_shape))
    # # test_tr = test.reshape((1, 128, 128))
    # recon = recon.detach()
    # make_visible(recon).save("recon32_32__aug_k_1.jpg")








    # print(f"TEST IMAGE SHAPE {test.shape} -- {recon.shape}")
    # grid = make_grid([test_tr, recon])
    # img = T.ToPILImage()(grid)
    # img.show()
