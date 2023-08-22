from tkinter import N
import torch.nn as nn
import torch
import copy
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from myNetwork import *
from torch.utils.data import DataLoader
import random
from Fed_utils import *
from proxy_data import *


class proxyServer:
    def __init__(self, device, learning_rate, numclass, feature_extractor, encode_model, test_transform,args):
        super(proxyServer, self).__init__()
        if args.encode=='lenet':
            self.Iteration = 250
        else:
            self.Iteration = 150
        self.learning_rate = learning_rate
        if args.dataset == 'cifar100':
            self.model = network(numclass, feature_extractor, 4)
        elif args.dataset == 'tiny_imagenet':
            self.model = network(numclass, feature_extractor, 8)
        else:
            self.model = network(numclass, feature_extractor, 11)
        self.encode_model = encode_model
        self.monitor_dataset = Proxy_Data(test_transform)
        self.new_set = []
        self.new_set_label = []
        self.numclass = 0
        self.device = device
        self.num_image = 20
        self.pool_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0
        self.args=args
        self.cur_perf=0
        self.ep_g=0


    def dataloader(self, pool_grad,model):
        self.pool_grad = pool_grad
        if len(pool_grad) != 0:
            self.reconstruction()
            self.monitor_dataset.getTestData(self.new_set, self.new_set_label)
            self.monitor_loader = DataLoader(dataset=self.monitor_dataset, shuffle=True, batch_size=64, drop_last=True)
            self.last_perf = 0
            self.best_model_1 = self.best_model_2
        if self.model.radius==0:
            self.cur_perf = self.monitor()
        else:
            self.cur_perf = self.f_monitor()
        if self.ep_g%self.args.tasks_global<self.args.proxy_init:
            self.cur_perf=0
        print(self.cur_perf)
        if self.cur_perf >= self.best_perf:
            self.best_perf = self.cur_perf
            self.best_model_2 = copy.deepcopy(self.model)

    def model_back(self):
        return [self.best_model_1, self.best_model_2]

    def monitor(self):
        self.model.eval()
        correct, total = 0, 0
        for step, (imgs, labels) in enumerate(self.monitor_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        
        return accuracy

    def f_monitor(self):
        features=[]
        labs=[]
        self.model.eval()
        correct, total = 0, 0
        for step, (imgs, labels) in enumerate(self.monitor_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                feature = self.model.feature_extractor(imgs)
            if step == 0:
                features = feature
                labs=labels
            else:
                features = torch.cat((features, feature), 0)
                labs = torch.cat((labs, labels), 0)
        features=features.detach().cpu().numpy()
        labs=labs.cpu().numpy()
        labels_set = np.unique(labs)
        cov_sum=[]
        proto_aug = []
        proto_aug_label = []
        covs=np.zeros((200,512))
        cov_sum=np.zeros(200)
        for i in labels_set:
            index=np.where(i==labs)[0]
            feature_classwise = features[index]
            cov=np.cov(feature_classwise.T)
            cov=np.square(np.diagonal(cov))
            covs[i]=cov
            cov_sum[i]=cov.sum()
        for i in range(features.shape[0]):
            num=0
            index=labs[i]
            while(num<5):
                noise= np.random.normal(0, covs[index], 512) * self.model.radius
                a=np.linalg.norm(noise)
                b=cov_sum[index]
                if np.square(a) > 0.1*cov_sum[index]:
                    continue
                else:
                    temp=features[i]+noise
                    proto_aug.append(temp)
                    proto_aug_label.append(index)
                    num+=1
        proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
        proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
        outputs=self.model.fc(proto_aug)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == proto_aug_label.cpu()).sum()
        total += len(proto_aug_label)
        accuracy = 100 * correct / total

        return accuracy

    def gradient2label(self):
        pool_label = []
        for w_single in self.pool_grad:
            pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            pool_label.append(pred.item())

        return pool_label

    def reconstruction(self):
        self.new_set, self.new_set_label = [], []

        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        pool_label = self.gradient2label()
        pool_label = np.array(pool_label)
        class_ratio = np.zeros((1, 200))

        for i in pool_label:
            class_ratio[0, i] += 1

        for label_i in range(100):
            if class_ratio[0, label_i] > 0:
                num_augmentation = self.num_image
                augmentation = []
                
                grad_index = np.where(pool_label == label_i)
                for j in range(len(grad_index[0])):
                    grad_truth_temp = self.pool_grad[grad_index[0][j]]

                    dummy_data = torch.randn((1, 3, self.args.img_size, self.args.img_size)).to(self.device).requires_grad_(True)
                    label_pred = torch.Tensor([label_i]).long().to(self.device).requires_grad_(False)

                    optimizer = torch.optim.LBFGS([dummy_data, ], lr=0.1)
                    criterion = nn.CrossEntropyLoss().to(self.device)

                    recon_model = copy.deepcopy(self.encode_model)
                    recon_model = model_to_device(recon_model, False, self.device)

                    for iters in range(self.Iteration):
                        def closure():
                            optimizer.zero_grad()
                            pred = recon_model(dummy_data)
                            dummy_loss = criterion(pred, label_pred)

                            dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True)

                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, grad_truth_temp):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        optimizer.step(closure)
                        current_loss = closure().item()

                        if iters == self.Iteration - 1:
                            print(current_loss)

                        if iters >= self.Iteration - self.num_image:
                            dummy_data_temp = np.asarray(tp(dummy_data.clone().squeeze(0).cpu()))
                            augmentation.append(dummy_data_temp)

                self.new_set.append(augmentation)
                self.new_set_label.append(label_i)


    