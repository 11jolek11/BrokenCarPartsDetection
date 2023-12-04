import os
import shutil
from datetime import datetime

import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from door_data import DoorsDataset2, DoorsDataset3, door_transforms
from src.settings import DEVICE

folder = "../../../logs/runs/"
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
summary_writer = SummaryWriter("../../../logs/runs/")


class RBM(nn.Module):
    def __init__(self, number_of_visible, number_of_hidden, k=33, *args, **kwargs):
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
        return self.number_of_hidden, self.number_of_visible

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

def train(model, data_loader, lr, epochs_number: int, optimizer, *args, **kwargs):
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs_number):
        total_loss_by_epoch = 0

        for batch in data_loader:
            batch = batch.reshape((-1, model.number_of_visible))
            batch = batch.to(DEVICE)

            v = model.forward(batch).to(DEVICE)

            loss = torch.mean(model.loss(batch)) - torch.mean(model.loss(v))

            print(f"Loss: {loss}")
            total_loss_by_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        summary_writer.add_scalar("Loss", total_loss_by_epoch, epoch)
        print('Epoch [{}/{}] --> loss: {:.4f}'.format(epoch + 1, epochs_number, total_loss_by_epoch))
        now = datetime.now()
        torch.save({
            "epoch": epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss_by_epoch,
        }, f"../../../checkpoint/checkpoint_RBM_{DEVICE.type}_{now.strftime('%m_%d_%Y_%H_%M_%S')}.pth")

    torch.save({
        "epochs_number": epochs_number,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"../../../checkpoint/RBM_{DEVICE.type}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.pth")


summary_writer.close()

if __name__ == "__main__":
    datas = DoorsDataset3(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
        transform=door_transforms
    )
    train_loader = DataLoader(dataset=datas, batch_size=1, shuffle=True)

    # my_model = RBM(128*128, 128*128+600, k=33)
    # train(my_model, train_loader, 0.001, 30, torch.optim.SGD)

    my_model = RBM(128 * 128, 128 * 128 + 600, k=3)
    train(my_model, train_loader, 0.001, 1, torch.optim.SGD)

    datas_test = DoorsDataset3(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
        transform=door_transforms
    )

    test = datas_test[0]
    make_visible = T.ToPILImage()
    make_visible(test).show()
    test = test.reshape((-1, my_model.number_of_visible))
    test = test.to(DEVICE)
    recon = my_model.forward(test)
    recon = recon.reshape((1, 128, 128))
    # test_tr = test.reshape((1, 128, 128))
    recon = recon.detach()
    make_visible(recon).show()
    # print(f"TEST IMAGE SHAPE {test.shape} -- {recon.shape}")
    # grid = make_grid([test_tr, recon])
    # img = T.ToPILImage()(grid)
    # img.show()
