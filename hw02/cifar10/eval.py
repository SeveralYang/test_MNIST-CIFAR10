import cv2
import numpy
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize
from model import test_cifar10_model
test_set = CIFAR10(root="../dataset", train=False, transform=Compose([
    Resize((256,256)),
    ToTensor()
]), download=True)
test_size = len(test_set)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)

model = test_cifar10_model()
model.load_state_dict(torch.load("../weight/cifar10_17.pth"))

back = Compose([
    Resize((448, 448)),
    ToPILImage()
])

print('---开始测试---')
# 不进行梯度调优
with torch.no_grad():
    total_loss = 0
    correct = 0
    for img, target in test_loader:
        img_cuda = img.cuda()
        target_cuda = target.cuda()
        output = model(img)
        """for i in range(64):
            img_test = img[i, :, :, :]
            img_test = back(img_test)       
            cv2.imshow(f"id:{i},label:{target[i]},predict{output[i].argmax()}", numpy.array(img_test))
            cv2.waitKey()
            cv2.destroyAllWindows()"""
        correct += (output.argmax(1) == target).sum()
        print(correct)
print('---测试完成，本轮正确率{}---'.format(correct / test_size))
