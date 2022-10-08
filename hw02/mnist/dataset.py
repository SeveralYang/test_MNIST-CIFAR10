from torchvision.datasets import MNIST

dataset = MNIST(root="../dataset",
                train=True,
                transform=None,
                download=True, )

n = len(dataset)
print(f" MNIST一共有{n}张图片")

for i in range(n):
    img, label = dataset.__getitem__(i)
    print(img, label)
    if i == 10:
        break
print("可知MNIST数据集中图像均为28x28的灰度图像")
