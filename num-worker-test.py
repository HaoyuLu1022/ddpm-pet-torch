from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms
 
 
transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
 
trainset = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,  #如果为True，从 training.pt 创建数据，否则从 test.pt 创建数据。
    download=True, #如果为true，则从 Internet 下载数据集并将其放在根目录中。 如果已下载数据集，则不会再次下载。
    transform=transform
)
 
print(f"num of CPU: {mp.cpu_count()}")
for num_workers in range(2, mp.cpu_count(), 2):  
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=num_workers, batch_size=64, pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

"""
A6000
num of CPU: 32
Finish with:8.508655786514282 second, num_workers=2
Finish with:4.447881698608398 second, num_workers=4
Finish with:3.1664600372314453 second, num_workers=6
Finish with:2.4951705932617188 second, num_workers=8
Finish with:2.0793204307556152 second, num_workers=10
Finish with:1.9567067623138428 second, num_workers=12
Finish with:1.9402432441711426 second, num_workers=14
Finish with:2.0018696784973145 second, num_workers=16
Finish with:1.9051053524017334 second, num_workers=18
Finish with:1.866199016571045 second, num_workers=20
Finish with:1.7788496017456055 second, num_workers=22
Finish with:1.919997215270996 second, num_workers=24
Finish with:1.9662563800811768 second, num_workers=26
Finish with:1.9567828178405762 second, num_workers=28
Finish with:1.9589223861694336 second, num_workers=30

4090
num of CPU: 32
Finish with:11.957906246185303 second, num_workers=2
Finish with:6.1901209354400635 second, num_workers=4
Finish with:4.504554986953735 second, num_workers=6
Finish with:3.542226791381836 second, num_workers=8
Finish with:3.0297322273254395 second, num_workers=10
Finish with:2.8556365966796875 second, num_workers=12
Finish with:2.836229085922241 second, num_workers=14
Finish with:2.9378437995910645 second, num_workers=16
Finish with:2.872610330581665 second, num_workers=18
Finish with:3.0367791652679443 second, num_workers=20
Finish with:3.0908493995666504 second, num_workers=22
Finish with:3.088261604309082 second, num_workers=24
Finish with:3.126222848892212 second, num_workers=26
Finish with:3.199321746826172 second, num_workers=28
"""
