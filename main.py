import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch
from model import Mymodel
import torch.nn as nn
import warnings
import numpy as np
plt.rcParams['font.sans-serif'] = ['YouYuan']
plt.rcParams['axes.unicode_minus'] = False

class Mydata(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx, :].reshape(1, -1)
        y = self.y[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


###模型的训练
def train(model, writer, device, train_dataloader, optimizer, loss_fn, epoch):
    model.train()
    train_loss = 0
    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('The loss of the model on the train dataset:{}'.format(train_loss))
    writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=epoch)


###模型的测试
def test(model, writer, device, test_dataloader, loss_fn, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target)

    print('The loss of the model on the test dataset:{}'.format(test_loss))
    writer.add_scalar(tag='test_loss', scalar_value=test_loss, global_step=epoch)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    input_df = pd.read_excel('./data.xlsx', sheet_name='input', header=None)
    output_df = pd.read_excel('./data.xlsx', sheet_name='output', header=None)
    input_test_df = pd.read_excel('./data.xlsx', sheet_name='input_test', header=None)
    output_test_df = pd.read_excel('./data.xlsx', sheet_name='output_test', header=None)
    train_dataset = Mydata(X=input_df.values, y=output_df.values)
    test_dataset = Mydata(X=input_test_df.values, y=output_test_df.values)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

    ###将过程结果写入tensorboard
    writer = SummaryWriter('logs')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###神经网络的训练和测试
    model = Mymodel()
    writer.add_graph(model=model, input_to_model=torch.randn(8, 1, 4))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    loss_fn.to(device)
    EPOCH = 200
    for epoch in range(EPOCH):
        print('EPOCH**************************{}/{}'.format(epoch + 1, EPOCH))
        train(model=model, writer=writer, device=device, train_dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn, epoch=epoch)
        test_acc = test(model=model, writer=writer, device=device, test_dataloader=test_dataloader, loss_fn=loss_fn, epoch=epoch)
    writer.close()

    ###绘制预测值和真实值对比曲线
    y_true=np.array(output_test_df)
    X_test=torch.reshape(torch.tensor(input_test_df.values,dtype=torch.float32),(-1,1,4))
    y_pred=torch.reshape(model(X_test),(-1,1)).detach().numpy()

    plt.plot(y_true,label='真实值')
    plt.plot(y_pred,label='预测值')
    plt.legend()
    plt.show(block=True)