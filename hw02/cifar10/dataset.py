from torchvision.datasets import CIFAR10

dataset = CIFAR10(root="../dataset",
                  train=True,
                  transform=None,
                  download=True, )

n = len(dataset)
print(f" CIFAR10一共有{n}张图片")

for i in range(n):
    img, label = dataset.__getitem__(i)
    print(img, label)
    if i == 10:
        break
print("可知CIFAR10数据集中图像均为32x32的RGB图像")
