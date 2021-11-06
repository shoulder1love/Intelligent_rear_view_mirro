# # import torch
# # import numpy as np
# #
# # data = [1,2,3,4,5,6]
# # x = np.array(data)
# # print(torch.cuda.is_available())
# #
# #
# # print(torch.__version__)
# # print(x)
#
# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import numpy as np
#
# data = [1,2,3,4,5,6]
# x = np.array(data)
# print(torch.cuda.is_available())
#
#
# print(torch.__version__)
# print(x)
# #fake
# x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100,1)
# x = Variable(x) # Variable
# x_np = x.data.numpy() #Torch的数据不能被plt识别， Varibale的数据存放在 .data 中
#
#
# y_relu = torch.relu(x).data.numpy()
# # y_sigmoid = F.sigmoid(x).data.numpy()
# # y_tahn = F.tanh(x).data.numpy()
# # y_softplus = F.softplus(x).data.numpy()
#
# #画图
# plt.figure(1, figsize=(8, 6))
# plt.subplot(221)
# plt.plot(x_np, y_relu, c='green', label='relu')
# plt.ylim((-1, 5))
# plt.legend(loc='best')
# plt.show()
import torch
from  torch.autograd import  Variable
import  torch.nn.functional as F
import matplotlib.pyplot as plt

# make fake data
n_data = torch.ones(100, 2)

# torch.normal(means, std, out=None)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               #标签为0  # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                #标签为1  # class1 y data (tensor), shape=(100, 1)

# 合并
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape(100, 1)
# y= x.pow(2) + 0.2*torch.rand(x.size())  # noise y data (tensor), shape(100, 1)

x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden_2, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden_2)
        self.predict = torch.nn.Linear(n_hidden_2, n_output)
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.hidden2(x))
        x = self.predict(x)
        return x
net = Net(2, 10, 10, 2) #(2(输入有x,y),10,2(输出有两类))
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# loss_func = torch.nn.MSELoss() # for regression
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
#训练
for t in range(100):
    out = net(x)

    loss = loss_func(out, y) # y = label, out = prediction

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #动态图
    # if t % 2 ==0:
    #     plt.cla()
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
    #     plt.pause(0.1)

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1] #[1]代表位置[0]代表数字结果
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()



