
import os.path
import pathlib
import PySide2.QtCore
import PySide2.QtUiTools
import PySide2.QtWidgets
import PySide2.QtGui
import threading
# import tensorboardX
# import torchvision.transforms
import torch.utils.data
import classify
import mynet

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
'''如果有显卡，通过显卡运行，否则通过cpu运行。'''
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print('device:',device)

class event_f(PySide2.QtCore.QObject):
    def __init__(self):
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == train_w.gui:
            if event.type() == PySide2.QtCore.QEvent.Close:
                obj.closeEvent()


class train_w(PySide2.QtCore.QObject):   #训练和使用网络
    def __init__(self, net, trainloader, testloader, b_s):
        super(train_w, self).__init__()
        self.gui = PySide2.QtUiTools.QUiLoader().load('./GUI/train_use.ui')
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader


        self.net.signals.text_train_ifo.connect(self.new_train_acc)         #设置处理信号的函数
        self.net.signals.text_test_ifo.connect(self.new_test_acc)
        self.net.signals.set_pgb_ifo.connect(self.set_pgb)
        self.net.signals.upd_pgb_ifo.connect(self.upd_pgb)

        self.gui.quit_b.setVisible(False)
        self.gui.progressBar.setVisible(False)

        self.gui.cramera.clicked.connect(self.camera)
        self.gui.use_net.clicked.connect(self.c_f_p)
        self.gui.get_path.clicked.connect(self._get_c_path)
        self.gui.train_b.clicked.connect(self.train)
        self.gui.quit_b.clicked.connect(self.quit)
        self.gui.test.clicked.connect(self.try_test)
        self.gui.installEventFilter(self)                   #加载信号过滤器

    def new_train_acc(self,epoch,acc,loss):         #通过信号更新数据
        self.gui.epoch.setText('epoch:'+ str(epoch))
        self.gui.train_acc.setText('train acc:' + str(acc))
        self.gui.loss.setText('loss:' + str(loss))

    def new_test_acc(self,acc):                     #通过信号更新数据
        self.gui.test_acc.setText('test acc:' + str(acc))

    def set_pgb(self,tol):
        self.gui.progressBar.setRange(0,tol)

    def upd_pgb(self,i):
        self.gui.progressBar.setValue(i)

    def try_test(self):
        self.net.test(test_dataloader= self.testloader)

    def train_start(self):  #训练网络
        self.net.quit = False
        self.net.net_train(train_dataloader= self.trainloader, test_dataloader= self.testloader)

    def quit(self):   #中止训练
        self.net.quit = True
        self.gui.train_b.setVisible(True)
        self.gui.quit_b.setVisible(False)
        self._threading_train.join()

    def camera(self):  #调用摄像头
        classify.opencamera(net= self.net.net, device= device)

    def c_f_p(self):   #分类文件夹中文件
        for i in range(self.gui.Lay.count()):           #清楚之前的图片
            self.gui.Lay.itemAt(i).widget().deleteLater()
        if  pathlib.Path(self.gui.fold.text()).is_dir():
            self._img_list, self._lab_list = classify.use_net_by_fold(net= self.net.net, device= device, ues_folder= self.gui.fold.text())
            for i in range(len(self._img_list)):
                lab_img = PySide2.QtWidgets.QLabel(self.gui)        #图片lab
                lab_label = PySide2.QtWidgets.QLabel(self.gui)      #标签lab
                lab_label.setText(classes[self._lab_list[i]])
                pix = PySide2.QtGui.QPixmap(self._img_list[i])
                lab_img.setPixmap(pix)
                self.gui.Lay.addWidget(lab_img)                     #加入scroolarea的layout中
                self.gui.Lay.addWidget(lab_label)

    def train(self):                        #在新线程中训练
        self._threading_train = threading.Thread(target= self.train_start)
        self.gui.progressBar.setVisible(True)
        self.gui.train_b.setVisible(False)
        self.gui.quit_b.setVisible(True)
        self._threading_train.start()

    def _get_c_path(self):  #选择文件夹
        self._filePath_c = PySide2.QtWidgets.QFileDialog.getExistingDirectory(self.gui, "选择文件夹")
        self.gui.fold.setText(self._filePath_c)

    def eventFilter(self, obj, event):                  #过滤event
        if obj is self.gui:
            if event.type() == PySide2.QtCore.QEvent.Close:
                self.net.quit = True                    #按下X后将训练线程关闭释放资源
                print(self.net.quit)
                self.gui.close()
                return True
        return super(train_w, self).eventFilter(obj, event)


class Start:

    def __init__(self):
        # qfile_start = PySide2.QtCore.QFile('D:\py_workspace\my_net_for_c10\GUI\untitled.ui')
        # qfile_start.open(PySide2.QtCore.QFile.ReadOnly)
        # qfile_start.close()

        self.gui = PySide2.QtUiTools.QUiLoader().load('./GUI/untitled.ui')
        self._net_type = 'Resnet18'
        # self._LR = 0.1
        # self._Mom = 0.9
        # self._gam = 0.9
        # self._L2 = 0.0005
        self._ckpt_l_p = 'None'
        self._start_w_c = False
        self._cifarx = 10

        self.gui.ck_path.setVisible(False)
        self.gui.getpath.setVisible(False)
        self.gui.l_p_lab.setVisible(False)

        self.gui.netchoose.currentIndexChanged.connect(self._choosenet) #更换网络
        self.gui.optimizer.currentIndexChanged.connect(self._chooseopt) #更换优化器
        self.gui.buttonGroup.buttonClicked.connect(self._if_new)        #是否创建新网络
        self.gui.buttonGroup2.buttonClicked.connect(self._cifar)        #选择cifar
        self.gui.create_net.clicked.connect(self._get_net)              #生成网络
        self.gui.getpath.clicked.connect(self._get_l_path)              #选择加载存档地址
        self.gui.ckpt_s_p_b.clicked.connect(self._get_s_path)           #选择保存地址
        self._optimizer = self.gui.optimizer.currentText()
        self.gui.cf10.setVisible(False)
        self.gui.cf100.setVisible(False)

    def _choosenet(self):  #更换网络
        self._net_type = self.gui.netchoose.currentText()
        #print(self._net_type)

    def _chooseopt(self):
        self._optimizer = self.gui.optimizer.currentText()
        if self._optimizer == 'SGD':
            self.gui.LR.setValue(0.1)
            self.gui.Mom.setValue(0.9)
            self.gui.gam.setValue(0.9)
            self.gui.L2.setValue(0.0005)
        elif self._optimizer == 'RMSProp':
            self.gui.LR.setValue(0.01)
            self.gui.Mom.setValue(0)
            self.gui.gam.setValue(1)
            self.gui.L2.setValue(0)
        elif self._optimizer == 'Adam':
            self.gui.LR.setValue(0.001)
            self.gui.Mom.setValue(0)
            self.gui.gam.setValue(1)
            self.gui.L2.setValue(0)
        elif self._optimizer == 'Adamax':
            self.gui.LR.setValue(0.002)
            self.gui.Mom.setValue(0)
            self.gui.gam.setValue(1)
            self.gui.L2.setValue(0)
        elif self._optimizer == 'SparseAdam':
            self.gui.LR.setValue(0.002)
            self.gui.Mom.setValue(0)
            self.gui.gam.setValue(1)
            self.gui.L2.setValue(0)

    def _if_new(self):      #是否创建新网络
        self._if_text = self.gui.buttonGroup.checkedButton().text()
        if self._if_text == '新建网络':
            self._start_w_c = False
            self.gui.ck_path.setVisible(False)
            self.gui.getpath.setVisible(False)
            self.gui.l_p_lab.setVisible(False)
        else:
            self._start_w_c = True
            self.gui.ck_path.setVisible(True)
            self.gui.getpath.setVisible(True)
            self.gui.l_p_lab.setVisible(True)

    def _cifar(self):  #选择cifar
        if self.gui.buttonGroup2.checkedButton().text() == 'cifar10':
            self._cifarx = 10
            #print(10)
        elif self.gui.buttonGroup2.checkedButton().text() == 'cifar100':
            self._cifarx = 100
            #print(100)

    def _get_l_path(self):  #选择加载存档地址
        self._filePath_l, _ = PySide2.QtWidgets.QFileDialog.getOpenFileName(
            self.gui,  # 父窗口对象
            "选择保存的参数文件",  # 标题
            r"./",  # 起始目录
            "选择保存的参数文件 (*.pth)"  # 选择类型过滤项，过滤内容在括号中
        )
        self.gui.ck_path.setText(self._filePath_l)

    def _get_s_path(self):  #选择保存地址
        self._filePath_s = PySide2.QtWidgets.QFileDialog.getExistingDirectory(self.gui, "选择存储路径")
        self.gui.ck_path_s.setText(self._filePath_s)

    def _get_net(self):  #生成网络

        self._LR = self.gui.LR.value()          #获取网络参数
        self._Mom = self.gui.Mom.value()
        self._L2 = self.gui.L2.value()
        self._gam = self.gui.gam.value()
        self._check_num = self.gui.num.value()
        self._max_num = self.gui.max_num.value()
        self._ckpt_s_p = self.gui.ck_path_s.text()
        self._ckpt_name = self.gui.s_name.text()
        self._batch_size = self.gui.ba_size.value()
        if self._cifarx == 10:
            self.train_dataloader, self.test_dataloader = mynet.get_cifar10_d(self._batch_size)
        elif self._cifarx == 100:
            self.train_dataloader, self.test_dataloader = mynet.get_cifar100_d(self._batch_size)

        if self._start_w_c:
            self._ckpt_l_p = self.gui.ck_path.text()
        if self._net_type == 'Resnet18':
            self.net = mynet.net_resnet18(device= device, optimizer_name= self._optimizer,
                                          LR= self._LR, Mom= self._Mom, L2= self._L2, gam= self._gam,
                                          check_num= self._check_num, max_epoch= self._max_num,
                                          ckpt_l_p= self._ckpt_l_p, ckpt_s_p= self._ckpt_s_p,
                                          checkpoint_name= self._ckpt_name, forcifar= self._cifarx)
        elif self._net_type == 'Resnet34':
            self.net = mynet.net_resnet34(device=device, optimizer_name= self._optimizer,
                                          LR=self._LR, Mom=self._Mom, L2=self._L2, gam=self._gam,
                                          check_num=self._check_num, max_epoch=self._max_num,
                                          ckpt_l_p=self._ckpt_l_p, ckpt_s_p=self._ckpt_s_p,
                                          checkpoint_name= self._ckpt_name, forcifar= self._cifarx)

        elif self._net_type == 'Resnet50':
            self.net = mynet.net_resnet50(device=device, optimizer_name= self._optimizer,
                                          LR=self._LR, Mom=self._Mom, L2=self._L2, gam=self._gam,
                                          check_num=self._check_num, max_epoch=self._max_num,
                                          ckpt_l_p=self._ckpt_l_p, ckpt_s_p=self._ckpt_s_p,
                                          checkpoint_name= self._ckpt_name, forcifar= self._cifarx)

        elif self._net_type == 'Resnet101':
            self.net = mynet.net_resnet101(device=device, optimizer_name= self._optimizer,
                                          LR=self._LR, Mom=self._Mom, L2=self._L2, gam=self._gam,
                                          check_num=self._check_num, max_epoch=self._max_num,
                                          ckpt_l_p=self._ckpt_l_p, ckpt_s_p=self._ckpt_s_p,
                                          checkpoint_name= self._ckpt_name, forcifar= self._cifarx)

        elif self._net_type == 'Resnet152':
            self.net = mynet.net_resnet152(device=device, optimizer_name= self._optimizer,
                                          LR=self._LR, Mom=self._Mom, L2=self._L2, gam=self._gam,
                                          check_num=self._check_num, max_epoch=self._max_num,
                                          ckpt_l_p=self._ckpt_l_p, ckpt_s_p=self._ckpt_s_p,
                                          checkpoint_name= self._ckpt_name, forcifar= self._cifarx)

        self.win2 = train_w(net= self.net, trainloader= self.train_dataloader, testloader= self.test_dataloader, b_s= self._batch_size)
        self.win2.gui.show()
        self.gui.close()



def main():
    # train_dataloader, test_dataloader = mynet.get_cifar10_d(batch_size= 128) #获取cifar10数据集
    # net = mynet.net_resnet18(device= device,start_with_checkpoint= True) #创建网络
    # # net.net_train(train_dataloader= train_dataloader, test_dataloader= test_dataloader)
    # #classify.use_net_by_fold(net= net.net,device= net.device)
    # net.test(test_dataloader= test_dataloader)
    # classify.opencamera(net= net.net, device= device)
    app = PySide2.QtWidgets.QApplication([])
    start = Start()
    start.gui.show()
    app.exec_()


if __name__ == '__main__':
    main()



