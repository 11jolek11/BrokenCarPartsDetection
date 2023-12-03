import cv2
import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from src.settings import DEVICE
from torch.utils.data import DataLoader
from door_data import DoorsDataset2, door_transforms
import torchvision.transforms as T
from PIL import Image


summary_writer = SummaryWriter("../../../logs/runs/")


class RBM(nn.Module):
    def __init__(self, number_of_visible, number_of_hidden, k=33, *args, **kwargs):
        super(RBM, self).__init__()
        self._number_of_visible = number_of_visible
        self._number_of_hidden = number_of_hidden
        self._k = k
        self.weight = nn.Parameter(torch.rand(number_of_hidden, number_of_visible)*1e-3)
        self.bias_v = nn.Parameter(torch.rand(1, number_of_visible)*1e-3)  # Bias for visible lay.
        self.bias_h = nn.Parameter(torch.rand(1, number_of_hidden)*1e-3)  # Bias for hidden lay.
        self.register_parameter("bias v", self.bias_v)
        self.register_parameter("bias h", self.bias_h)
        self.register_parameter("weights", self.weight)

    def get_properties(self):
        return self._number_of_hidden, self._number_of_visible

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
            batch = batch.reshape((-1, 1024))
            batch = batch.to(DEVICE)

            v = model.forward(batch).to(DEVICE)

            loss = torch.mean(model.loss(batch)) - torch.mean(model.loss(v))

            print(f"Loss: {loss}")
            total_loss_by_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        summary_writer.add_scalar("loss", total_loss_by_epoch, epoch)
        print('Epoch [{}/{}] --> loss: {:.4f}'.format(epoch + 1, epochs_number, total_loss_by_epoch))


summary_writer.close()

if __name__ == "__main__":
    datas = DoorsDataset2(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/JPEGImages",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
        transform=door_transforms
    )
    train_loader = DataLoader(dataset=datas, batch_size=1, shuffle=True)

    my_model = RBM(32*32, 32*32*2, k=33*9)
    train(my_model, train_loader, 0.001, 80, torch.optim.SGD)

    datas_test = DoorsDataset2(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/JPEGImages",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
        transform=door_transforms
    )

    test = datas_test[0]
    print(f"TEST IMAGE SHAPE {test.shape}")
    make_visible = T.ToPILImage()
    make_visible(test).show()
    test = test.reshape((-1, 1024))
    test = test.to(DEVICE)
    recon = my_model.forward(test)
    recon = recon.reshape((32, 32))
    make_visible(recon).show()
