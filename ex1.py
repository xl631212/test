# pytorch exercise for regression
# Author: Xuying LI
# Date: 2018/10/15
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt 


# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.2* torch.rand(x.size())

x ,y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 继承torch中的模块
class Net(torch.nn.Module):

	# 初始化设定,定义层
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = torch.nn.Linear(n_feature, n_hidden) # (输入个数，输出个数)
		self.predict = torch.nn.Linear(n_hidden, n_output) 

	# 搭图
	def forward(self, x):
		x = F.relu(self.hidden(x))	
		x = self.predict(x) # 预测的时候不使用激励函数
		return x

net = Net(1, 10, 1) #(输入特征个数，神经元， 输出个数)
print(net)


#定义优化器和损失函数
optimizer = torch.optim.SGD(net.parameters(), lr = 0.3) #（）
loss_func = torch.nn.MSELoss()

# 定义可视化过程
plt.ion()

for t in range(1000):
	prediction = net(x)

	loss = loss_func(prediction, y)

	# 优化器的初始化， 向后传输梯度， 下一步
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if t % 5 == 0:
		plt.cla()
		plt.scatter(x.data.numpy(), y.data.numpy())
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
		plt.pause(0.1)

plt.ioff()
plt.show()
