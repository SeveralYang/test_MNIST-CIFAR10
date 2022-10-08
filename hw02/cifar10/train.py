import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import test_cifar10_model

writer = SummaryWriter("../log")

train_set = CIFAR10(root="../dataset", train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([256, 256]),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
]), download=True)

test_set = CIFAR10(root="../dataset", train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([256, 256]),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
]), download=True)
test_size = len(test_set)
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)
model = test_cifar10_model()
# model.load_state_dict(torch.load('../weight/weight.pth'))
model = model.cuda()

loss_function = nn.CrossEntropyLoss().cuda()

learn_rate = 0.00001
optimizer = Adam(model.parameters(), lr=learn_rate)

total_train = 0
epoch = 500

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
    writer.add_scalar('test_correct', correct / test_size, i)
    print('---第{}轮测试完成，本轮总误差{},正确率{}---'.format(i + 1, total_loss, correct / test_size))

    # 保存训练模型
    torch.save(model.parameters(), '../weight/cifar10_{}.pth'.format(i))

writer.close()
