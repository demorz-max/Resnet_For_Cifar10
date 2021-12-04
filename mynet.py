import os
import pathlib

import tensorboardX
import torch
import torchvision
import torch.utils.data
import PySide2.QtCore

RGB_mean = (0.4914, 0.4822, 0.4465)  # 数据集的RGB平均值
RGB_std = (0.2023, 0.1994, 0.2010)  # 数据集RGB的方差
classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # 分类

train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((251, 251)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(RGB_mean, RGB_std)
    ]
)
# 对训练集图像的预处理

test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(RGB_mean, RGB_std)
    ]
)  # 对测试集的图像的预处理，暂时设为和训练集相同

class mysignals(PySide2.QtCore.QObject):
    text_train_ifo = PySide2.QtCore.Signal(float,float,float)
    text_test_ifo = PySide2.QtCore.Signal(float)
    upd_pgb_ifo = PySide2.QtCore.Signal(int)
    set_pgb_ifo = PySide2.QtCore.Signal(int)

# 获取cifar10数据包
def get_cifar10_d(batch_size):
    print('#加载数据#')
    train_data_root = './cifar10data'  # 训练集地址
    test_data_root = './cifar10data'  # 测试集地址
    train_dataset = torchvision.datasets.CIFAR10(root=train_data_root,
                                                 train=True, download=True,
                                                 transform=train_transform)  # 加载训练集，如果对应地址没有则下载
    test_datasett = torchvision.datasets.CIFAR10(root=test_data_root,
                                                 train=False,
                                                 download=True,
                                                 transform=test_transform)  # 加载测试集，如果对应地址没有则下载

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=True)  # 构建可迭代的数据装载器
    test_dataloader = torch.utils.data.DataLoader(dataset=test_datasett,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False)

    return train_dataloader, test_dataloader

def get_cifar100_d(batch_size):
    print('#加载数据#')
    train_data_root = './cifar100data'  # 训练集地址
    test_data_root = './cifar100data'  # 测试集地址
    train_dataset = torchvision.datasets.CIFAR100(root=train_data_root,
                                                 train=True, download=True,
                                                 transform=train_transform)  # 加载训练集，如果对应地址没有则下载
    test_datasett = torchvision.datasets.CIFAR100(root=test_data_root,
                                                 train=False,
                                                 download=True,
                                                 transform=test_transform)  # 加载测试集，如果对应地址没有则下载

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=True)  # 构建可迭代的数据装载器
    test_dataloader = torch.utils.data.DataLoader(dataset=test_datasett,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False)

    return train_dataloader, test_dataloader

# 获取网络和操作器
class net_resnet18:
    def __init__(self, device,
                 LR=0.1,  # 学习率
                 Mom=0.9,  # 动量因子
                 L2=5e-4,  # 权重衰减
                 gam=0.9,  # gamma值
                 start_with_checkpoint=True,
                 checkpoint_name= 'ckpt',
                 ckpt_s_p = '.\checkpoint',  # 参数保存文件夹
                 ckpt_l_p = '.\checkpoint\ckpt.pth', #加载文件
                 check_num=5,  # 每多少轮保存一次
                 max_epoch=50,
                 forcifar= 10,
                 optimizer_name='SGD'
                 ):

        print('#构建resnet18网络模型#')
        net = torchvision.models.resnet18()
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, forcifar)  # 重写全连接层
        net.to(device)  # 使用对应设备
        criterion = torch.nn.CrossEntropyLoss()  # 设置损失函数
        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=net.parameters(),
                                        lr=LR,
                                        momentum=Mom,
                                        weight_decay=L2)  # 设置优化器

        if optimizer_name == 'RMSProp':
            optimizer = torch.optim.RMSprop(params=net.parameters(),
                                            lr=LR,
                                            momentum=Mom,
                                            eps=1e-8,
                                            alpha=0.99,
                                            weight_decay=L2)

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params=net.parameters(),
                                         lr=LR,
                                         betas=(0.9, 0.999),
                                         eps=1e-8,
                                         weight_decay=L2)

        if optimizer_name == 'Adamax':
            optimizer = torch.optim.Adamax(params=net.parameters(),
                                           lr=LR,
                                           betas=(0.9, 0.999),
                                           eps=1e-8,
                                           weight_decay=L2)

        if optimizer_name == 'SparseAdam':
            optimizer = torch.optim.SparseAdam(params=net.parameters(),
                                               lr=LR,
                                               betas=(0.9, 0.999),
                                               eps=1e-8)
        Explr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gam)  # 绑定学习率衰减器

        self.net = net
        self.criterion = criterion
        self._optimizer = optimizer
        self._Explr = Explr
        self._ckptpath = ckpt_s_p
        self.startepoch = 0
        self.device = device
        self._writer = tensorboardX.SummaryWriter()
        self._check_num = check_num
        self.maxepoch = max_epoch
        #self._checkpoint_name = checkpoint_name
        self._ckptfile_path = os.path.join(ckpt_s_p, checkpoint_name + '.pth')
        self._ckpt_l_p = ckpt_l_p
        self.quit = False
        self.signals = mysignals()
        # 读取网络参数
        if start_with_checkpoint and pathlib.Path(self._ckpt_l_p).is_file() and self._ckpt_l_p.endswith('.pth'):
            print('从', self._ckpt_l_p, '加载网络参数')
            checkpoint = torch.load(self._ckpt_l_p,map_location= self.device)
            net.load_state_dict(checkpoint['net'])
            self.startepoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('新网络')

    def train(self, train_dataloader, epoch):
        print('\nepoch:', epoch)
        self.net.train()
        loss_mean = 0
        correct = 0
        total = 0
        self.signals.set_pgb_ifo.emit(len(train_dataloader))
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)  # 启用显卡

            # 前向
            outputs = self.net(inputs)

            # 后向
            self._optimizer.zero_grad()  # 梯度清零
            loss = self.criterion(outputs, labels)
            loss.backward()

            # 更新权重
            self._optimizer.step()

            # 统计分类情况
            __, prdictrd = outputs.max(1)  # 计算分类结果
            total += labels.size(0)  # 统计训练总数
            correct += prdictrd.eq(labels).sum().item()
            loss_mean += loss.item()
            self.signals.upd_pgb_ifo.emit(i)
            if self.quit:
                break
        if i != 0:
            loss_mean = loss_mean / i  # !!!!!!!!!!!!!!!!!!!!!!!!
        acc = correct / total
        self.train_acc = acc
        self.loss = loss_mean
        self.signals.text_train_ifo.emit(epoch,acc,loss_mean)
        print('loss_mean:', loss_mean)
        print('acc:', acc)
        self._writer.add_scalar(tag='train_loss_mean',
                               scalar_value=loss_mean,
                               global_step=epoch)
        self._writer.add_scalar(tag='train_acc',
                               scalar_value=acc,
                               global_step=epoch)

    def test(self, test_dataloader, epoch = 0, if_writer=False):
        print('\ntest:')
        self.net.eval()
        loss_mean = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)  # 启用显卡

                # 基本同上
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                # 统计分类情况
                __, prdictrd = outputs.max(1)  # 计算分类结果
                total += labels.size(0)  # 统计训练总数
                correct += prdictrd.eq(labels).sum().item()
                loss_mean += loss.item()
                if self.quit:
                    break
            loss_mean = loss_mean / i  # !!!!!!!!!!!!!!!!!!!!!!!!
            acc = correct / total
            self.test_acc = acc
            print('loss_mean:', loss_mean)
            print('acc:', acc)
            self.signals.text_test_ifo.emit(acc)
            if if_writer:
                self._writer.add_scalar(tag='test_loss_mean',
                                       scalar_value=loss_mean,
                                       global_step=epoch)
                self._writer.add_scalar(tag='test_acc',
                                       scalar_value=acc,
                                       global_step=epoch)

    # 训练网络
    def net_train(self, train_dataloader, test_dataloader):
        self.net.eval()
        with torch.no_grad():
            self._writer.add_graph(model=self.net, input_to_model=torch.rand(1, 3, 224, 224).to(self.device))  # 绘制网络结构图
        # 开始训练网络
        for epoch in range(self.startepoch + 1, self.maxepoch + 1):
            self.train(train_dataloader=train_dataloader, epoch=epoch)
            if epoch % self._check_num == 0:  # 一定轮数后保存网络
                self.test(test_dataloader=test_dataloader, epoch=int(epoch / self._check_num), if_writer= True)
                print('Saving..')
                state = {
                    'net': self.net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                    'epoch': epoch,
                }
                if not pathlib.Path(self._ckptpath).is_dir():
                    os.mkdir(self._ckptpath)
                torch.save(state, self._ckptfile_path)  # !!!!!!!!!!!!!!!!!!!!!!!!
            self._writer.add_scalar(tag='lr',
                                   scalar_value=int(self._optimizer.state_dict()['param_groups'][0]['lr']),
                                   global_step=epoch)
            self._Explr.step()
            if self.quit:
                break
        print('训练结束')
        self.startepoch = epoch

class net_resnet34(net_resnet18):
    def __init__(self, device,
                 LR=0.1,  # 学习率
                 Mom=0.9,  # 动量因子
                 L2=5e-4,  # 权重衰减
                 gam=0.9,  # gamma值
                 start_with_checkpoint=True,
                 checkpoint_name='ckpt',
                 ckpt_s_p='.\checkpoint',  # 参数保存文件夹
                 ckpt_l_p='.\checkpoint\ckpt.pth',  # 加载文件
                 check_num=5,  # 每多少轮保存一次
                 max_epoch=50,
                 forcifar=10,
                 optimizer_name= 'SGD'
                 ):
        start_epoch = 0
        print('#构建resnet34网络模型#')
        net = torchvision.models.resnet34()
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, forcifar)  # 重写全连接层
        net.to(device)  # 使用对应设备
        criterion = torch.nn.CrossEntropyLoss()  # 设置损失函数
        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=net.parameters(),
                                        lr=LR,
                                        momentum=Mom,
                                        weight_decay=L2)  # 设置优化器


        if optimizer_name == 'RMSProp':
            optimizer = torch.optim.RMSprop(params= net.parameters(),
                                            lr= LR,
                                            momentum= Mom,
                                            eps= 1e-8,
                                            alpha= 0.99,
                                            weight_decay= L2)


        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params= net.parameters(),
                                         lr= LR,
                                         betas= (0.9,0.999),
                                         eps= 1e-8,
                                         weight_decay= L2)

        if optimizer_name == 'Adamax':
            optimizer = torch.optim.Adamax(params= net.parameters(),
                                           lr= LR,
                                           betas= (0.9,0.999),
                                           eps= 1e-8,
                                           weight_decay= L2)

        if optimizer_name == 'SparseAdam':
            optimizer = torch.optim.SparseAdam(params= net.parameters(),
                                               lr= LR,
                                               betas= (0.9,0.999),
                                               eps= 1e-8)

        Explr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gam)  # 绑定学习率衰减器\


        self.net = net
        self.criterion = criterion
        self._optimizer = optimizer
        self._Explr = Explr
        self._ckptpath = ckpt_s_p
        self._startepoch = start_epoch
        self.device = device
        self._writer = tensorboardX.SummaryWriter()
        self._check_num = check_num
        self.maxepoch = max_epoch
        # self._checkpoint_name = checkpoint_name
        self._ckptfile_path = os.path.join(ckpt_s_p, checkpoint_name, '.pth')
        self._ckpt_l_p = ckpt_l_p
        # 读取网络参数
        if start_with_checkpoint and pathlib.Path(self._ckpt_l_p).is_file() and self._ckpt_l_p.endswith('.pth'):
            print('从', self._ckpt_l_p, '加载网络参数')
            checkpoint = torch.load(self._ckpt_l_p)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('新网络')

class net_resnet50(net_resnet18):
    def __init__(self, device,
                 LR=0.1,  # 学习率
                 Mom=0.9,  # 动量因子
                 L2=5e-4,  # 权重衰减
                 gam=0.9,  # gamma值
                 start_with_checkpoint=True,
                 checkpoint_name='ckpt',
                 ckpt_s_p='.\checkpoint',  # 参数保存文件夹
                 ckpt_l_p='.\checkpoint\ckpt.pth',  # 加载文件
                 check_num=5,  # 每多少轮保存一次
                 max_epoch=50,
                 forcifar=10,
                 optimizer_name='SGD'
                 ):
        start_epoch = 0
        print('#构建resnet50网络模型#')
        net = torchvision.models.resnet50()
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, forcifar)  # 重写全连接层
        net.to(device)  # 使用对应设备
        criterion = torch.nn.CrossEntropyLoss()  # 设置损失函数
        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=net.parameters(),
                                        lr=LR,
                                        momentum=Mom,
                                        weight_decay=L2)  # 设置优化器

        if optimizer_name == 'RMSProp':
            optimizer = torch.optim.RMSprop(params=net.parameters(),
                                            lr=LR,
                                            momentum=Mom,
                                            eps=1e-8,
                                            alpha=0.99,
                                            weight_decay=L2)

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params=net.parameters(),
                                         lr=LR,
                                         betas=(0.9, 0.999),
                                         eps=1e-8,
                                         weight_decay=L2)

        if optimizer_name == 'Adamax':
            optimizer = torch.optim.Adamax(params=net.parameters(),
                                           lr=LR,
                                           betas=(0.9, 0.999),
                                           eps=1e-8,
                                           weight_decay=L2)

        if optimizer_name == 'SparseAdam':
            optimizer = torch.optim.SparseAdam(params=net.parameters(),
                                               lr=LR,
                                               betas=(0.9, 0.999),
                                               eps=1e-8)
        Explr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gam)  # 绑定学习率衰减器

        self.net = net
        self.criterion = criterion
        self._optimizer = optimizer
        self._Explr = Explr
        self._ckptpath = ckpt_s_p
        self._startepoch = start_epoch
        self.device = device
        self._writer = tensorboardX.SummaryWriter()
        self._check_num = check_num
        self.maxepoch = max_epoch
        # self._checkpoint_name = checkpoint_name
        self._ckptfile_path = os.path.join(ckpt_s_p, checkpoint_name, '.pth')
        self._ckpt_l_p = ckpt_l_p
        # 读取网络参数
        if start_with_checkpoint and pathlib.Path(self._ckpt_l_p).is_file() and self._ckpt_l_p.endswith('.pth'):
            print('从', self._ckpt_l_p, '加载网络参数')
            checkpoint = torch.load(self._ckpt_l_p)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('新网络')

class net_resnet101(net_resnet18):
    def __init__(self, device,
                 LR=0.1,  # 学习率
                 Mom=0.9,  # 动量因子
                 L2=5e-4,  # 权重衰减
                 gam=0.9,  # gamma值
                 start_with_checkpoint=True,
                 checkpoint_name='ckpt',
                 ckpt_s_p='.\checkpoint',  # 参数保存文件夹
                 ckpt_l_p='.\checkpoint\ckpt.pth',  # 加载文件
                 check_num=5,  # 每多少轮保存一次
                 max_epoch=50,
                 forcifar=10,
                 optimizer_name='SGD'
                 ):
        start_epoch = 0
        print('#构建resnet101网络模型#')
        net = torchvision.models.resnet101()
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, forcifar)  # 重写全连接层
        net.to(device)  # 使用对应设备
        criterion = torch.nn.CrossEntropyLoss()  # 设置损失函数
        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=net.parameters(),
                                        lr=LR,
                                        momentum=Mom,
                                        weight_decay=L2)  # 设置优化器

        if optimizer_name == 'RMSProp':
            optimizer = torch.optim.RMSprop(params=net.parameters(),
                                            lr=LR,
                                            momentum=Mom,
                                            eps=1e-8,
                                            alpha=0.99,
                                            weight_decay=L2)

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params=net.parameters(),
                                         lr=LR,
                                         betas=(0.9, 0.999),
                                         eps=1e-8,
                                         weight_decay=L2)

        if optimizer_name == 'Adamax':
            optimizer = torch.optim.Adamax(params=net.parameters(),
                                           lr=LR,
                                           betas=(0.9, 0.999),
                                           eps=1e-8,
                                           weight_decay=L2)

        if optimizer_name == 'SparseAdam':
            optimizer = torch.optim.SparseAdam(params=net.parameters(),
                                               lr=LR,
                                               betas=(0.9, 0.999),
                                               eps=1e-8)
        Explr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gam)  # 绑定学习率衰减器

        self.net = net
        self.criterion = criterion
        self._optimizer = optimizer
        self._Explr = Explr
        self._ckptpath = ckpt_s_p
        self._startepoch = start_epoch
        self.device = device
        self._writer = tensorboardX.SummaryWriter()
        self._check_num = check_num
        self.maxepoch = max_epoch
        # self._checkpoint_name = checkpoint_name
        self._ckptfile_path = os.path.join(ckpt_s_p, checkpoint_name, '.pth')
        self._ckpt_l_p = ckpt_l_p
        # 读取网络参数
        if start_with_checkpoint and pathlib.Path(self._ckpt_l_p).is_file() and self._ckpt_l_p.endswith('.pth'):
            print('从', self._ckpt_l_p, '加载网络参数')
            checkpoint = torch.load(self._ckpt_l_p)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('新网络')

class net_resnet152(net_resnet18):
    def __init__(self, device,
                 LR=0.1,  # 学习率
                 Mom=0.9,  # 动量因子
                 L2=5e-4,  # 权重衰减
                 gam=0.9,  # gamma值
                 start_with_checkpoint=True,
                 checkpoint_name='ckpt',
                 ckpt_s_p='.\checkpoint',  # 参数保存文件夹
                 ckpt_l_p='.\checkpoint\ckpt.pth',  # 加载文件
                 check_num=5,  # 每多少轮保存一次
                 max_epoch=50,
                 forcifar=10,
                 optimizer_name='SGD'
                 ):
        start_epoch = 0
        print('#构建resnet152网络模型#')
        net = torchvision.models.resnet152()
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, forcifar)  # 重写全连接层
        net.to(device)  # 使用对应设备
        criterion = torch.nn.CrossEntropyLoss()  # 设置损失函数
        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=net.parameters(),
                                        lr=LR,
                                        momentum=Mom,
                                        weight_decay=L2)  # 设置优化器

        if optimizer_name == 'RMSProp':
            optimizer = torch.optim.RMSprop(params=net.parameters(),
                                            lr=LR,
                                            momentum=Mom,
                                            eps=1e-8,
                                            alpha=0.99,
                                            weight_decay=L2)

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params=net.parameters(),
                                         lr=LR,
                                         betas=(0.9, 0.999),
                                         eps=1e-8,
                                         weight_decay=L2)

        if optimizer_name == 'Adamax':
            optimizer = torch.optim.Adamax(params=net.parameters(),
                                           lr=LR,
                                           betas=(0.9, 0.999),
                                           eps=1e-8,
                                           weight_decay=L2)

        if optimizer_name == 'SparseAdam':
            optimizer = torch.optim.SparseAdam(params=net.parameters(),
                                               lr=LR,
                                               betas=(0.9, 0.999),
                                               eps=1e-8)
        Explr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gam)  # 绑定学习率衰减器

        self.net = net
        self.criterion = criterion
        self._optimizer = optimizer
        self._Explr = Explr
        self._ckptpath = ckpt_s_p
        self._startepoch = start_epoch
        self.device = device
        self._writer = tensorboardX.SummaryWriter()
        self._check_num = check_num
        self.maxepoch = max_epoch
        # self._checkpoint_name = checkpoint_name
        self._ckptfile_path = os.path.join(ckpt_s_p, checkpoint_name, '.pth')
        self._ckpt_l_p = ckpt_l_p
        # 读取网络参数
        if start_with_checkpoint and pathlib.Path(self._ckpt_l_p).is_file() and self._ckpt_l_p.endswith('.pth'):
            print('从', self._ckpt_l_p, '加载网络参数')
            checkpoint = torch.load(self._ckpt_l_p)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('新网络')