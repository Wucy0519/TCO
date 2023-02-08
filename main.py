import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from torchvision import models
from utils.weight_init import weight_init_kaiming
from model.model import get_model
from dataset.dataset import get_data 
import shelve
from wcy_method import Tco
      
def save_checkpoint(state, is_best,model_name,dataset_name):
    filename="Save/"
    cute = model_name+'_'+dataset_name+".pth"
    torch.save(state, filename+"normal/"+cute)
    if is_best:
        torch.save(state, filename+"best/"+cute)
        
def _accuracy(net,test_loader):
    net.train(False)
    num_total = 0
    num_acc = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            output = net(imgs)
            _, pred = torch.max(output, 1)
            num_acc += torch.sum(pred == labels.detach_())
            num_total += labels.size(0)
    LV = num_acc.detach().cpu().numpy() * 100 / num_total
    return LV
        
#Run this script on GPUs 
GPUS = min(1,int(torch.cuda.device_count()) )
#Add the dataset name you want to train to this list
#You can choose to join or remove models or datasets that you want to test in this list
dataset_list = ["cifar100","VOCdevkit","CUB"]
#Add the model name you want to train to this list
#If you want to train vit (the default is TransFG), download the pretrained model, add it to "pretrained" fold and name it "vit.npz"
model_list = ["ViT","DenseNet169","Resnet50"]

#Change batch size for each Model
bs_list = {"DenseNet169":128,"ViT":32,"Resnet50":128}

#Your dataset path. Such as : /home/wcy/DataSet
path = '/XXX/XXX'

epochs = 75
base_lr = GPUS*1e-3
momentum = 0.9
weight_decay = 1e-4
img_size = 224
step_size = 25
gamma = 0.1

#Random drop optimizer parameter
p_drop = 0.8

#The confidence threshold
s_l = 0.7

#The switch of TCO method. If you want use TCO to train your model, turn it into True
tco_switch = True 

print('Training process starts:...')
if GPUS > 1:
   print('More than one GPU are used...')
           
for dataset_name in dataset_list:
    for model_name in model_list:

        batch_size = bs_list[model_name]
        
        print('*' * 25)
        print("batch_size :    ",batch_size,"      |")
        print("Training Dataset: ",dataset_name,", and Model: ",model_name)
        print('*' * 25)
        print('Epoch\tTrainLoss\tTrainAcc\tTestAcc')
        print('-' * 50)
        
        train_loader,test_loader = get_data(Path = path, img_size = img_size, batch_size = batch_size, dataset_name = dataset_name)
        
        Wcynet = (get_model(model_name,dataset_name)).cuda()
        criterion = nn.CrossEntropyLoss()
        solver = torch.optim.SGD(Wcynet.parameters(), lr=base_lr, momentum=momentum,weight_decay=weight_decay)
        schedule = torch.optim.lr_scheduler.StepLR(solver, step_size=step_size, gamma=gamma)
        if tco_switch:
            #doubleSGD: Random Drop Optimizer transformed form SGD
            Tco_mode = Tco(net = Wcynet,test_loader = test_loader,drop_p = p_drop,s_l = s_l,gamma = gamma,steps = step_size,lr = base_lr,weight_decay = weight_decay,momentum=momentum,solver_name="doubleSGD")
        else:
            print("Not use TCO method in your training")
        
        epochss = np.arange(1,epochs + 1)
        test_acc = list()
        train_acc = list()
        
        best_acc = 0.0
        Wcynet.train(True)
        for epoch in range(epochs):
           if  tco_switch:
               Wcynet = Tco_mode(Wcynet)
           
           num_correct = 0
           train_loss_epoch = list()
           num_total = 0
          
           for imgs, labels in train_loader:
               solver.zero_grad()
               imgs = imgs.cuda()
               labels = labels.cuda()
               
               output = Wcynet(imgs)
               loss = criterion(output, labels)
               _, pred = torch.max(output, 1)
               num_correct += torch.sum(pred == labels.detach_())
               num_total += labels.size(0)
               train_loss_epoch.append(loss.item())
               loss.backward()
               solver.step()
           train_acc_epoch = num_correct.detach().cpu().numpy() * 100 / num_total
           avg_train_loss_epoch = sum(train_loss_epoch) / len(train_loss_epoch)
           test_acc_epoch = _accuracy(Wcynet,test_loader)
           schedule.step()
           is_best = test_acc_epoch >= best_acc
           
           if is_best:
               print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%'.format(epoch + 1, avg_train_loss_epoch, train_acc_epoch, test_acc_epoch),end='')
               print("||-|-||This result is better||-|-||")
           else:
               print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%'.format(epoch + 1, avg_train_loss_epoch, train_acc_epoch, test_acc_epoch))
           best_acc = max(test_acc_epoch, best_acc)
                                           
        print("Finish!!!")
