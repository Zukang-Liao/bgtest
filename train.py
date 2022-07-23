# For training CNN models
# Output: a trained model (mid.pth) at args.SAVE_DIR

import os
import torch
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split
from model import VGGnet, SimpleNet, ViT, resnet18
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from sampler import imbalanceSampler, orderedSampler
from datasets import BgChallengeDB, LeakyDataset, NoisyDataset
from utils import get_config
from triplet_loss import TripletLoss
# import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != 'cpu':
    gpus = [0, 1, 2]
res_mean = torch.tensor([0.4717, 0.4499, 0.3837])
res_std = torch.tensor([0.2600, 0.2516, 0.2575])
outputSize = 224
vit_arch = 'vit_base_patch16_224_in21k'
triplet_lambda = 0.1 # loss + lambda * triplet_loss
nb_devices = torch.cuda.device_count()


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ConfigPath', type=str, default="./config.yaml")
    parser.add_argument("--SAVE_DIR", type=str,default="./saved_models")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--nThreads", type=int, default=0)
    parser.add_argument('--pretrain', action='store_true', default=False) # whether to save plots
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--mid", type=str, default="-1") # model id
    parser.add_argument("--aug", type=str, default="") # augmentation: 'r' -- rotate, 'b' -- brightness, 's' -- size, or '' -- none
    parser.add_argument("--opt", type=str, default="adam") # optimiser: 'adam', 'sgd', 'rms' (RMSprop)
    parser.add_argument("--dbmode", type=str, default="bgtest") # 'bgtest': no aug, 'onlyfg': use only foreground to train
    parser.add_argument("--onlyfg_rate", type=float, default=0.5)


    # Params for anomalies
    # propotion of data used for training
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--target_class", type=str, default="")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.0)

    # vgg13bn or simple for CNN5
    parser.add_argument("--arch", type=str, default="vgg13bn")
    parser.add_argument("--adv", type=bool, default=False)
    parser.add_argument("--anomaly", type=str, default="")

    args = parser.parse_args()
    args.SAVE_PATH = os.path.join(args.SAVE_DIR, f"{args.mid}.pth")
    args.LOG_DIR = os.path.join(args.SAVE_DIR, "logs")
    args.LOG_PATH = os.path.join(args.LOG_DIR, str(args.mid))
    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)
    args.max_aug = 15 if args.aug == 'r' else 0.3
    return args


def verify_args(args):
    """
    Anomalies:
        "1": ordered batch (do not shuffle after each epoch)
        "2": noisy data -- include purely random noise as images in training
        "3": imbalanced data -- remove part of examples of a targeted class
        "4": ordered training -- feed CNNs with examples of class 1 and then class 2 in order (Not in use in this work)
        "5": impaired data -- train CNNs with only a small portion of training set.
        "6": impaired augmentation -- remove some degrees from augmentation (e.g. do not augment the image with 5 degrees) 
        "7": impaired labels -- wrong labels
        "8": data leakage
    """
    assert args.anomaly != '4', "Anomaly type 4 is not yet ready and it was not used in the project"
    if args.anomaly == "3":
        assert 0 < args.r < 1, "Please specify ratio: args.r"
        assert args.target_class != "", "Please specify targeted class: args.target_class"
    elif args.anomaly == "5":
        assert 0 < args.r < 1, "Please specify ratio: args.r"
        assert args.target_class == "", "Please set the anomaly type to be 3 for a targeted class"
    elif args.anomaly == "2" or args.anomaly == "7":
        assert 0 < args.noise < 1, "Please specify noise level"
    elif args.anomaly == "8":
        assert 0 < args.r < 1, "Please specify ratio: args.r"
    elif args.anomaly == "6":
        assert args.aug != "", "Please specify augmentation scheme for type 6 anomaly"
    if args.adv:
        assert 0 < args.epsilon < 1, "Please specify epsilon for adversarial training"


class AnomalyAug:
    """Rotate by one of the given angles."""
    def __init__(self, max_aug, aug_type):
        self.max_aug = max_aug
        self.aug_type = aug_type
        if aug_type == "r":
            self.target_augs = np.random.choice(range(-5, 5+1), 2, replace=False)
        elif aug_type == "s":
            self.target_augs = (0.95, 1.05)
        elif aug_type == "b":
            self.target_augs = (0.95, 1.05)

    def __call__(self, x):
        if self.aug_type == "r":
            angle = (np.random.random() - 0.5) * 2 * self.max_aug
            while np.round(angle) in self.target_augs:
                angle = (np.random.random() - 0.5) * 2 * self.max_aug
            return TF.rotate(x, angle)
        elif self.aug_type == "s":
            sv = (np.random.random() - 0.5) * 2 * self.max_aug + 1
            while self.target_augs[0] < sv < self.target_augs[1]:
                sv = (np.random.random() - 0.5) * 2 * self.max_aug + 1
            return TF.affine(x, scale=sv, angle=0, translate=[0,0], shear=0)
        elif self.aug_type == "b":
            bv = (np.random.random() - 0.5) * 2 * self.max_aug + 1
            while self.target_augs[0] < bv < self.target_augs[1]:
                bv = (np.random.random() - 0.5) * 2 * self.max_aug + 1
            return TF.adjust_brightness(x, bv)


def get_transform(args, split):
    if split == 'train':
        if args.aug == "r":
            # rotation
            _transform = AnomalyAug(args.max_aug, args.aug) if args.anomaly=="6" else transforms.RandomRotation(args.max_aug)
        elif args.aug == "s":
            # scaling
            _transform = AnomalyAug(args.max_aug, args.aug) if args.anomaly=="6" else transforms.RandomAffine(degrees=0, scale=(1-args.max_aug, 1+args.max_aug))
        elif args.aug == "b":
            # brightness
            _transform = AnomalyAug(args.max_aug, args.aug) if args.anomaly=="6" else transforms.ColorJitter(brightness=args.max_aug)
        else:
            transform = transforms.Compose([
                transforms.Normalize(res_mean, res_std)
            ])
            return transform
        transform = transforms.Compose([
                        _transform,
                        transforms.Normalize(res_mean, res_std)
                    ])
        return transform
    else:
        transform = transforms.Compose([
            transforms.Normalize(res_mean, res_std)
        ])
    return transform


def get_dataGen(args, CONFIG, split):
    transforms = get_transform(args=args, split=split)
    if split == 'train':
        data =  BgChallengeDB(CONFIG['BGDB']['ORIGINAL_DIR'],
                         overlap='', 
                         TenCrop=False,
                         mode=args.dbmode, 
                         split=split, 
                         outputSize=outputSize,
                         seed=args.seed, 
                         r=args.r,
                         bgtransforms=transforms,
                         onlyfg_rate=args.onlyfg_rate)
        if args.anomaly == "8":
            testdata = BgChallengeDB(CONFIG['BGDB']['ORIGINAL_DIR'],
                                     overlap='', 
                                     TenCrop=False,
                                     mode='bgtest', 
                                     split='val', 
                                     outputSize=outputSize,
                                     seed=args.seed, 
                                     r=args.r)
            data = LeakyDataset(data, testdata, args.r, args.seed)
        if args.anomaly == "2" or args.anomaly == "7":
            data = NoisyDataset(data, args.noise, args.anomaly)
        shuffle = False if args.anomaly=="1" or args.anomaly=="4" else True
        if args.anomaly == "3":
            sampler = imbalanceSampler(data, int(args.target_class), args.r, args.batch_size, shuffle=True)
            dataGen = DataLoader(data, batch_sampler=sampler, num_workers=args.nThreads)
            return dataGen
        elif args.anomaly == "4":
            # ordered training: class by class
            sampler = orderedSampler(data, args.batch_size, nb_classes=CONFIG['BGDB']['num_class'], shuffle=True)
            dataGen = DataLoader(data, batch_sampler=sampler, num_workers=args.nThreads)
            return dataGen
        elif args.anomaly == "5":
            nb_data = int(len(data) * args.r)
            gen = torch.Generator().manual_seed(args.seed)
            data, _ = random_split(data, [nb_data, len(data)-nb_data], generator=gen)
    else:
        data =  BgChallengeDB(CONFIG['BGDB']['ORIGINAL_DIR'],
                         overlap='', 
                         TenCrop=False,
                         mode='bgtest', 
                         split=split, 
                         outputSize=outputSize,
                         seed=args.seed, 
                         r=args.r,
                         bgtransforms=transforms)
        shuffle = False
    dataGen = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.nThreads)
    return dataGen


# FGSM attack code
def fgsm_attack(args, image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, torch.min(image), torch.max(image))
    # Return the perturbed image
    return perturbed_image


def train(args, CONFIG):
    trainGen = get_dataGen(args, CONFIG, split='train')
    testGen = get_dataGen(args, CONFIG, split='val')
    writer = SummaryWriter(args.LOG_PATH)
    if 'vgg' in args.arch:
        net = VGGnet(args.arch, CONFIG['BGDB']['num_class'], args.pretrain)
    elif 'resnet' in args.arch:
        net = resnet18(args.arch, CONFIG['BGDB']['num_class'], args.pretrain)
    elif 'vit' == args.arch:
        net = ViT(vit_arch, CONFIG['BGDB']['num_class'], img_size=outputSize, pretrained=args.pretrain, drop_rate=0.1)
    else:
        net = SimpleNet()
        print("Training SimpleNet")
    if torch.cuda.device_count() > 1:
        print("DEVICE IS CUDA")
        net = torch.nn.DataParallel(net, device_ids=gpus)
        cudnn.benchmark = True
        net = net.to(device)
    if args.opt == 'adam':
        optimiser = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimiser = optim.SGD(net.parameters(), lr=args.lr)
    elif args.opt == 'rms':
        optimiser = optim.RMSprop(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    if args.dbmode == 'triplet':
        tripletloss = TripletLoss(device)
    test_loss_trace = []

    for e in range(args.epoch):
        net.train()
        running_loss = 0.
        nb_trainbatch = len(trainGen)
        for i, data in enumerate(trainGen):
            images, labels = data['img_data'], data['Label']
            # plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0))
            # plt.show()
            images, labels = images.to(device), labels.to(device)
            if args.adv:
                images.requires_grad = True
            optimiser.zero_grad()
            if args.dbmode != 'triplet':
                out = net(images)
                loss = criterion(out, labels)
            else:
                triplet_data = data['triplet_data']
                triplet_data = triplet_data.view(-1, triplet_data.shape[-3], outputSize, outputSize)
                # triplet_labels = np.repeat(labels.cpu().numpy(), 2)
                triplet_labels = np.repeat(np.array(range(len(labels))), 2)
                triplet_labels = torch.from_numpy(triplet_labels).to(device)
                if nb_devices == 0:
                    ins = net.inspect(triplet_data)
                else:
                    model = net.module
                    model = model.cuda()
                    triplet_data = triplet_data.to(device)
                    ins = model.inspect(triplet_data)
                out = ins["Linear_0"]
                loss = criterion(out[1::2], labels) # onlyfc and then original
                if args.arch == 'vit':
                    _tripletloss = tripletloss(torch.mean(nn.functional.normalize(ins["Act"], dim=1), dim=1), triplet_labels)
                else:
                    _tripletloss = tripletloss(nn.functional.normalize(ins["Act"], dim=1), triplet_labels)
                loss = loss + triplet_lambda * _tripletloss
            loss.backward()

            if args.adv:
                img_grad = images.grad.data
                # Call FGSM Attack
                perturbed_data = fgsm_attack(args, images, args.epsilon, img_grad)
                perturbed_out = net(perturbed_data)
                perturbed_loss = criterion(perturbed_out, labels)
                perturbed_loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if i % 10 == 0:
                print(f'Epoch: {e}, batch: {i} / {nb_trainbatch}, training loss: {running_loss/(i+1)}')
            # break # test
        running_loss /= len(trainGen)
        writer.add_scalar('TrainLoss', running_loss, e)       
        with torch.no_grad():
            test_loss = 0
            _correct, _total = 0, 0
            net.eval()
            for i, data in enumerate(testGen):
                images, labels = data['img_data'], data['Label']
                images, labels = images.to(device), labels.to(device)
                out = net(images)
                loss = criterion(out, labels)
                _, predictions = torch.max(out, axis=1)
                _correct += sum(predictions==labels).item()
                _total += labels.size(0)
                test_loss += loss.item()
                # break # test
            test_acc = _correct / _total
            test_loss /= len(testGen)
            test_loss_trace.append(test_loss)
            print("Epoch: %d, test_loss:%.3f, test_acc:%.3f" % (e+1, test_loss, test_acc))
        writer.add_scalar('TestLoss', test_loss, e)
        writer.add_scalar('TestAcc', test_acc, e)
        # if args.keep_steps and e+1 in args.steps:
        #     torch.save(net.state_dict(), os.path.join(args.SAVE_DIR, f"{args.mid}_{e+1}.pth"))
    torch.save(net.state_dict(), args.SAVE_PATH) # grad won't be saved
    log_model(args, test_acc)
    

def log_model(args, test_acc):
    comment = ''
    if args.target_class != '':
        comment += f"target_class: {args.target_class}"
    if args.noise > 0:
        comment += f"noise: {args.noise}"
    if args.epsilon > 0:
        comment += f"epsilon: {args.epsilon}"
    pretrain = args.pretrain if args.arch != "simple" else 'NA'
    aug = f"{args.aug}_{args.dbmode}"
    if args.dbmode == 'onlyfg':
        aug = f"{aug}{args.onlyfg_rate}"
    anomaly = 'NA' if args.anomaly == '' else args.anomaly
    with open(os.path.join(args.SAVE_DIR, "model_label.txt"), "a") as f:
        # path opt augmentation_info testacc pretrain epoch lr modelname adv_trained anomaly comment        
        f.write(f"{args.mid}.pth {args.opt} {aug} {test_acc:.3f} {pretrain} {args.epoch} {args.batch_size} {args.lr} {args.arch} {args.adv} {anomaly} {args.r} \"{comment}\"\n")


if __name__ == "__main__":
    args = argparser()
    CONFIG = get_config(args)
    verify_args(args)
    train(args, CONFIG)
