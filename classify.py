'''使用训练好的网络进行图像分类'''
import os
import PIL.Image
import torch.utils.data
import torchvision
import cv2

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
RGB_mean = (0.4914, 0.4822, 0.4465) #数据集的RGB平均值
RGB_std = (0.2023, 0.1994, 0.2010) #数据集RGB的方差
img_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(RGB_mean, RGB_std)
    ]
)

# class imgdataset(torch.utils.data.Dataset):
#     def __init__(self,path_list):
#         self.path_list = path_list
#     def __getitem__(self, item):
#         path_img = self.path_list[item]
#         img =

def img_p2tensor(img_path,transform):    #通过地址获取对应的tensor
    img = PIL.Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor

def get_jpeg_path(fold_path): #获取文件夹下的jepg文件,返回一个列表
    files = os.listdir(fold_path)
    img_names = list(filter(lambda x:x.endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.jpeg', '.pbm', '.ppm', '.tif', '.tiff')), files))
    img_path = []
    for i in range(len(img_names)):
        img_path.append(os.path.join(fold_path,img_names[i]))
    return img_path

def opencamera(net,device): #与摄像头交互
    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    net.eval()
    with torch.no_grad():
        while(True):
            ret, img = camera.read()
            if ret:
                imgRGB = PIL.Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                img_tensor =  img_transform(imgRGB)
                img_tensor = img_tensor.unsqueeze_(0)
                img_tensor = img_tensor.to(device)
                output = net(img_tensor)
                __, prdictrd = output.max(1)
                label = classes[int(prdictrd)]
                cv2.putText(img= img,text= label, org=(100,200), fontFace= cv2.FONT_HERSHEY_TRIPLEX, fontScale= 2.0, color=(200,100,200))
                cv2.imshow('camera', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        camera.release()
        cv2.destroyAllWindows()

def use_net_by_fold(net, device, ues_folder = 'F:\机器视觉\pic'): #分类文件夹中的文件
    img_list = get_jpeg_path(fold_path= ues_folder)
    the_end = classifyimage(path_list= img_list, device= device)
    outputs_lab = the_end.classify(net= net)
    #print(classes[int(outputs_lab)])
    return img_list, outputs_lab

class classifyimage: #用来分类图像,输入图像地址列表

    def __init__(self,path_list,device):
        self.path_list = path_list
        self.device = device

    def classify(self,net):
        labels = []
        net.eval()
        with torch.no_grad():
            for i in range(len(self.path_list)):
                img_tensor = img_p2tensor(self.path_list[i],img_transform)
                img_tensor = img_tensor.unsqueeze_(0)
                img_tensor = img_tensor.to(self.device)
                output = net(img_tensor)
                __, prdictrd = output.max(1)
                labels.append(int(prdictrd))
        return labels

