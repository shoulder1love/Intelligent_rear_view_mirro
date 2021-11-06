"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
"""
import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y)
# Data.DataLoader是对于batch训练的参数
loader = Data.DataLoader(       # 让训练变成一批一批的
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 训练是否每次重新打乱顺序 # random shuffle for training
    num_workers=2,              # 每次用多少线程 # subprocesses for loading data
)


def show_batch():
    for epoch in range(3):   #数据整体训练3次 # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()