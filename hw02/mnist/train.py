import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from model import test_mnist_model

writer = SummaryWriter("../log")

train_set = MNIST(root="../dataset", train=True, transform=ToTensor(), download=True)
test_set = MNIST(root="../dataset", train=False, transform=ToTensor(), download=True)
test_size = len(test_set)
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)
model = test_mnist_model()
"""dic = torch.load('../weight/weight.pth')
model.load_state_dict(dic)"""
model = model.cuda()

loss_function = nn.CrossEntropyLoss().cuda()

learn_rate = 0.00001
optimizer = Adam(model.parameters(), lr=learn_rate)

total_train = 0
epoch = 50

for i in range(epoch):
    print('---开始第{}轮训练---'.format(i + 1))
    for img, target in train_loader:
        img = img.cuda()
        target = target.cuda()
        output = model(img)
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train += 1

        print('训练次数{}，当前误差{}'.format(total_train + 1, loss.item()))
        writer.add_scalar('train_loss', loss.item(), total_train)

    print('---第{}轮训练完成，开始测试---'.format(i + 1))
    # 不进行梯度调优
    with torch.no_grad():
        total_loss = 0
        correct = 0
        for img, target in test_loader:
            img = img.cuda()
            target = target.cuda()
            output = model(img)
            loss = loss_function(output, target)
            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum()
    print('---第{}轮测试完成，本轮总误差{},正确率{}---'.format(i + 1, total_loss, correct / test_size))
    writer.add_scalar('test_loss', total_loss, i)
    # 保存训练模型
    # torch.save(model, '../weight/mnist_{}.pth'.format(i))

writer.close()
