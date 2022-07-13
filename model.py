import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from torchsummary import summary
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class VGGnet(nn.Module):
    def __init__(self, arch, nb_class, pretrained=False):
        super(VGGnet, self).__init__()
        # self.model = models.vgg13_bn(pretrained=True, progress=True)
        self.arch = arch
        self.nb_class = nb_class
        if '13' in arch:
            if 'bn' in arch:
                self.model = models.vgg13_bn(pretrained=pretrained, progress=True)
            else:
                self.model = models.vgg13(pretrained=pretrained, progress=True)
        elif '11' in arch:
            if 'bn' in arch:
                self.model = models.vgg11_bn(pretrained=pretrained, progress=True)
            else:
                self.model = models.vgg11(pretrained=pretrained, progress=True)


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model.classifier = None
        self.classifier = nn.Sequential(nn.Linear(512, nb_class))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.model.features(x)
        # x = self.model.avgpool(x)
        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

    def inspect(self, x):
        results = {}
        batch_size = x.size(0)
        # layer_idx = -1
        for i, f in enumerate(self.model.features):
            x = f(x)
            # if f._get_name() == "Conv2d":
                # layer_idx += 1
            # results[f._get_name()+f"_{layer_idx}"] = x
        results['Conv-1'] = x
        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        results["Act"] = x
        for j, f in enumerate(self.classifier):
            x = f(x)
            # results[f._get_name()+f"_{j}"] = x
        results['Linear_0'] = x
        return results


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout1 = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.pool(F.relu(self.dropout1(self.conv2(x))))
        x = F.relu(self.dropout1(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = self.fc2(x)
        return x

    def inspect(self, x):
        results = {}
        batch_size = x.size(0)
        results["Conv2d_0"] = self.conv1(x)
        results["Conv2d_1"] = self.conv2(self.pool(F.relu(results["Conv2d_0"])))
        results["Conv2d_2"] = self.conv3(self.pool(F.relu(results["Conv2d_1"])))
        x = torch.flatten(F.relu(results["Conv2d_2"]), 1)
        results["Linear_1"] = F.relu(self.fc1(x))
        results["Linear_0"] = self.fc2(results["Linear_1"])
        return results



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        t = self.shortcut(x)
        out += t
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=9):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        pre_out = x.view(x.size(0), -1)
        final = self.fc(pre_out)
        return final


    
    def inspect(self, x):
        results = {}
        # batch_size = x.size(0)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        results["Conv-1"] = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        results["Act"] = out
        y = self.fc(out)
        results["Linear_0"] = y
        return results
    
    def load(self, path="resnet_cifar10.pth"):
        tm = torch.load(path, map_location="cpu")        
        self.load_state_dict(tm)
        

def resnet18(arch, num_classes, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()

        in_sd = {}
        out_sd = {}
        for k in model_dict:
            if k in checkpoint:
                in_sd[k] = checkpoint[k]
            else:
                if 'shortcut' in k:
                    k_model = k.replace('shortcut', 'downsample')
                    if k_model in checkpoint:
                        in_sd[k] = checkpoint[k_model]
                        continue
                out_sd[k] = model_dict[k]

        del in_sd['fc.bias']
        del in_sd['fc.weight']
        model_dict.update(in_sd)
        model.load_state_dict(model_dict)
    return model


class ViT(nn.Module):
    def __init__(self, model_name, num_classes, img_size=224, pretrained=True, drop_rate=0.1):
        super(ViT, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.drop_rate = drop_rate
        self.pretrained = pretrained
        self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate = drop_rate,
                img_size = img_size
            )
        self.model.reset_classifier(num_classes)

    def forward(self, x):
        return self.model.forward(x)

    def inspect(self, x):
        results = {}
        out = self.model.forward_features(x)
        results["Conv-1"] = out
        results["Act"] = out
        y = self.model.get_classifier()(out)
        results["Linear_0"] = y
        return results
