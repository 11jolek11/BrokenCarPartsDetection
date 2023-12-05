import os
import shutil
from datetime import datetime
import uuid

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from door_data import DoorsDataset2, DoorsDataset3, door_transforms
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
        self.weight = nn.Parameter(torch.rand(number_of_hidden, number_of_visible)*1e-3)
        self.bias_v = nn.Parameter(torch.rand(1, number_of_visible)*1e-3)  # Bias for visible lay.
        self.bias_h = nn.Parameter(torch.rand(1, number_of_hidden)*1e-3)  # Bias for hidden lay.
        self.register_parameter("bias v", self.bias_v)
        self.register_parameter("bias h", self.bias_h)
        self.register_parameter("weights", self.weight)

    def get_properties(self):
        hiper_params = [self.number_of_hidden, self.number_of_visible, self._k]
        hiper_params_labels = ["hidden units", "visible units", "k"]
        return dict(zip(hiper_params_labels, hiper_params))

    def sample_h_from_v(self, v0):
        phv = torch.sigmoid(nn.functional.linear(v0, self.weight, self.bias_h))
        # h = nn.functional.relu(phv)
        h = torch.bernoulli(phv)
        return h, phv

    def sample_v_from_h(self, h0):
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

        for batch in data_loader:
            batch = batch.reshape((-1, model.number_of_visible))
            batch = batch.to(DEVICE)

            v = model.forward(batch).to(DEVICE)

            loss = torch.mean(model.loss(batch)) - torch.mean(model.loss(v))

            print(f"Loss: {loss} Epoch [{epoch + 1}/{epochs_number}]")
            total_loss_by_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summary_writer.add_graph(model, batch)

        summary_writer.add_scalar("Loss", total_loss_by_epoch, epoch)
        print('Epoch [{}/{}] --> loss: {:.4f}'.format(epoch + 1, epochs_number, total_loss_by_epoch))
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
    # summary_writer.add_hparams(hpams, {"min_loss_by_epoch": np.array(min(all_loss_by_epoch))}, run_name=train_uuid)
    print(f"Train series UUID {train_uuid}")
    # summary_writer
    summary_writer.close()


def test(model, data_loader, file_name):
    model = model.eval()
    model = model.to(DEVICE)

    list_img = []

    for batch in data_loader:
        original_batch = torch.clone(batch).reshape((1, 128, 128))
        original_batch = original_batch.to(DEVICE)
        batch = batch.reshape((-1, model.number_of_visible))
        batch = batch.to(DEVICE)

        v = model.forward(batch).to(DEVICE)
        v = v.reshape((1, 128, 128))
        list_img.append(original_batch)
        list_img.append(v)

    grid = make_grid(list_img, nrow=2)
    # grid.resize((len(list_img)/2, 1, 128, 128))
    # grid = make_grid(list_img)
    # summary_writer.add_images("Compare", grid)
    img = T.ToPILImage()(grid)
    img.save(file_name)

if __name__ == "__main__":
    datas = DoorsDataset3(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
        transform=door_transforms
    )

    train_loader = DataLoader(dataset=datas, batch_size=1, shuffle=True)

    # my_model = RBM(128*128, 128*128+600, k=33)
    # train(my_model, train_loader, 0.001, 30, torch.optim.SGD)

    # my_model = RBM(128 * 128, 800, k=3)
    # train(my_model, train_loader, 0.001, 19, torch.optim.SGD)

    # my_model = RBM(128 * 128, 128 * 128, k=3)
    # train(my_model, train_loader, 0.001, 1, torch.optim.SGD)

    my_model = RBM(128 * 128, 600, k=3)
    train(my_model, train_loader, 0.001, 6, datas.get_parts(), torch.optim.SGD)
    # 6

    test_data = DoorsDataset3(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/testset/annotations.json",
        transform=door_transforms
    )

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

    test(my_model, test_loader, "on_test.jpg")

    test(my_model, train_loader, "on_train.jpg")

    datas_test = DoorsDataset3(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
        transform=door_transforms
    )

    test = datas_test[0]
    make_visible = T.ToPILImage()
    make_visible(test).save("testttth.jpg")
    test = test.reshape((-1, my_model.number_of_visible))
    test = test.to(DEVICE)
    recon = my_model.forward(test)
    recon = recon.reshape((1, 128, 128))
    # test_tr = test.reshape((1, 128, 128))
    recon = recon.detach()
    make_visible(recon).save("recon.jpg")
    # print(f"TEST IMAGE SHAPE {test.shape} -- {recon.shape}")
    # grid = make_grid([test_tr, recon])
    # img = T.ToPILImage()(grid)
    # img.show()
