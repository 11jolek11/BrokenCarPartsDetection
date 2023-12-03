import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from src.settings import DEVICE
from torch.utils.data import DataLoader
from door_data import DoorsDataset, DoorsDataset2, door_transforms
# from src.utils.augumentator import p

summary_writer = SummaryWriter()


class RBM(nn.Module):
    def __init__(self, number_of_visible, number_of_hidden, k=33, *args, **kwargs):
        super(RBM, self).__init__()
        self._number_of_visible = number_of_visible
        self._number_of_hidden = number_of_hidden
        self._k = k
        self.weight = nn.Parameter(torch.rand(number_of_visible, number_of_hidden)*1e-3)
        self.bias_v = nn.Parameter(torch.rand(1, number_of_visible)*1e-3)  # Bias for visible lay.
        # print(f"Orginal bias_v type {type(self.bias_v)}")
        self.bias_h = nn.Parameter(torch.rand(1, number_of_hidden)*1e-3)  # Bias for hidden lay.
        self.register_parameter("bias v", self.bias_v)
        self.register_parameter("bias h", self.bias_h)
        self.register_parameter("weights", self.weight)
        # self.weight.to(DEVICE)
        # self.bias_v.to(DEVICE)
        # self.bias_h.to(DEVICE)

        # self.register_parameter("bias v", nn.Parameter(torch.rand(1, number_of_hidden)).to(DEVICE))
        #
        # self.register_parameter("bias h", nn.Parameter(torch.rand(1, number_of_visible)).to(DEVICE))
        #
        # self.register_parameter("weights", nn.Parameter(torch.rand(number_of_visible, number_of_hidden)).to(DEVICE))

    def get_properties(self):
        return self._number_of_hidden, self._number_of_visible

    def sample_h_from_v(self, v0):
        # print(f"v0 type {v0.dtype}, weight type {self.weight.dtype}, bias type {self.bias_h.dtype}")
        # print(f"v0 shape {v0.shape}, weight shape {self.weight.shape}, bias shape {self.bias_h.shape}")
        phv = torch.sigmoid(nn.functional.linear(v0, self.weight, self.bias_h))
        # h = torch.bernoulli(phv)
        h = nn.functional.relu(phv)
        # print(f"H nan: {torch.isnan(h).any()}, bias_h nan: {torch.isnan(self.bias_h).any()}")
        return h, phv

    def sample_v_from_h(self, h0):
        # print(f"h0 type {h0.dtype}, weight type {self.weight.dtype}, bias type {self.bias_h.dtype}")
        # print(f"h0 shape {h0.shape}, weight shape {self.weight.shape}, bias shape {self.bias_h.shape}")
        pvh = torch.sigmoid(nn.functional.linear(h0, self.weight.T, self.bias_v))
        # v = torch.bernoulli(pvh)
        v = nn.functional.relu(pvh)
        # print(f"V nan: {torch.isnan(v).any()}, bias_v nan: {torch.isnan(self.bias_v).any()}")
        return v, pvh

    def forward(self, v):
        h, phv = self.sample_h_from_v(v)
        # CD-k algo
        for i in range(self._k):
            v, pvh = self.sample_v_from_h(phv)
            h, phv = self.sample_h_from_v(v)
        v, pvh = self.sample_v_from_h(phv)
        # print(f"V nan: {torch.isnan(v).any()}, bias_v nan: {torch.isnan(self.bias_v).any()}")
        return v

    # TODO(11jolek11): Use reconstruction loss as a loss?
    # loss aka "free energy"
    def loss(self, v):
        # FIXME(11jolek11): why matrix is full of nan values?
        # print(f"V nan: {torch.isnan(v).any()}, bias_v nan: {torch.isnan(self.bias_v).any()}")
        vt = torch.matmul(v, torch.t(self.bias_v))
        # ht = torch.sum(torch.log(torch.exp(torch.matmul(v, self.weight) + self.bias_h) + 1), dim=1)
        temp = torch.log(torch.add(torch.exp(nn.functional.linear(v, self.weight, self.bias_h)), 1))
        # print(f"V nan: {torch.isnan(v).any()}, bias_v nan: {torch.isnan(self.bias_v).any()}")
        ht = torch.sum(temp, dim=1)
        # ht = torch.sum(torch.log(torch.add(torch.exp(nn.functional.linear(v, self.weight, self.bias_h)), 1)))


        return -vt - ht
        # print(f" v bias shape: {self.bias_v.shape} -- v shape {v.shape}")
        # vbias_term = v.mv(self.bias_v)
        # wx_b = nn.functional.linear(v, self.weight, self.bias_h)
        # hidden_term = wx_b.exp().add(1).log().sum(1)
        # return (-hidden_term - vbias_term).mean()



def train(model, data_loader, lr, epochs_number: int, optimizer, *args, **kwargs):
    # if not isinstance(optimizer, torch.optim.Optimizer):
    #     raise AttributeError("Wrong Optimizer")
    # try:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # except AttributeError as e:
    #     print(f'{e.name} - provide correct args for optimizer {optimizer.__class__.__name__}')
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs_number):
        total_loss_by_epoch = 0

        for batch in data_loader:
            # print(f"Batch size: {batch.shape}")
            # FIXME(11jolek11): Wrong tensor shape you dumb
            # batch = torch.flatten(batch)
            batch = batch.reshape((-1, 1024))
            # print(f"Flat batch size: {batch.shape}")
            batch = batch.to(DEVICE)



            v = model.forward(batch).to(DEVICE)

            loss = torch.mean(model.loss(batch)) - torch.mean(model.loss(v))
            # loss = loss.to(DEVICE)

            print(f"Loss: {loss}")
            total_loss_by_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        summary_writer.add_scalar("loss", total_loss_by_epoch, epoch)
        # print(f"Loss: {total_loss_by_epoch}")
        print('Epoch [{}/{}] --> loss: {:.4f}'.format(epoch + 1, epochs_number, total_loss_by_epoch))


# summary_writer.flush()
# summary_writer.close()

# 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
from torch.optim import Adam

class NN_Network(nn.Module):
    def __init__(self,in_dim,hid,out_dim):
        super(NN_Network, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid)
        self.linear2 = nn.Linear(hid, out_dim)
        # self.linear1.weight = torch.nn.Parameter(torch.zeros(in_dim, hid))
        # self.linear1.bias = torch.nn.Parameter(torch.ones(hid))
        # self.linear2.weight = torch.nn.Parameter(torch.zeros(in_dim, hid))
        # self.linear2.bias = torch.nn.Parameter(torch.ones(hid))

    def forward(self, input_array):
        h = self.linear1(input_array)
        y_pred = self.linear2(h)
        return y_pred



if __name__ == "__main__":
    from door_data import DoorsDataset, door_transforms

    # datas = DoorsDataset("C:/Users/dabro/PycharmProjects/scientificProject/notebooks/CarPartsDatasetExperimentDir/output", transform=door_transforms)
    datas = DoorsDataset2(
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/JPEGImages",
        "C:/Users/dabro/PycharmProjects/scientificProject/data/Car-Parts-Segmentation-master/Car-Parts-Segmentation-master/trainingset/annotations.json",
        transform=door_transforms
    )
    train_loader = DataLoader(dataset=datas, batch_size=1, shuffle=True)

    my_model = RBM(32*32, 32*32)
    train(my_model, train_loader, 0.001, 500, torch.optim.SGD)

    # in_d = 5
    # hidn = 2
    # out_d = 3
    # net = NN_Network(in_d, hidn, out_d)
    # print(list(net.parameters()))



