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

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape(100, 1)
y= x.pow(2) + 0.2*torch.rand(x.size())  # noise y data (tensor), shape(100, 1)

x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x
net = Net(1, 10, 1)
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

#训练
for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #动态图
    if t % 5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()




