import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import numpy as np

class network(nn.Module):

    def __init__(self, numclass, feature_extractor,avg):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.avgpool = nn.AvgPool2d(avg, stride=1)
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        self.transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.radius=0.15
        # classifier

    def forward(self, input):
        x = self.feature(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data 
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias


    def feature_extractor(self, inputs):
        x=self.feature(inputs)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def predict(self, fea_input):
        return self.fc(fea_input)
    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_radius(self, model,dataset, task_size, device,num_img):
        class_means=[]
        radius = []
        classes=list(range(task_size))
        for i in classes:
            images = dataset.get_image_class(i)
            x = self.Image_transform(images, self.transform).cuda(device)
            model.eval()
            for i in range(num_img):
                j = 50 * i
                imgs = x[j:j + 50]
                feature = model.feature_extractor(imgs)
                if i == 0:
                    features = feature
                else:
                    features = torch.cat((features, feature), 0)
                del feature
                features = features.detach().cpu().numpy()
                features = torch.from_numpy(features).to(device)
                torch.cuda.empty_cache()

            features = features.detach().cpu().numpy()
            feature_dim = features.shape[1]
            cov = np.cov(features.T)
            radius.append(np.trace(cov) / feature_dim)
        self.radius = np.sqrt(np.mean(radius))
        print('radius_')
        print(self.radius)





class LeNet(nn.Module):
    def __init__(self, padding,stride,channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=padding // 2, stride=stride),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=padding // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=padding // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())
