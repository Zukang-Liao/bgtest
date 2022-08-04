import torch
import os
from glob import glob
import numpy as np
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import imagenet_models
from torch.utils.data import random_split


class BgChallengeDB():
    
    def __init__(self, dbdir, TenCrop, mode, overlap='', split="val", outputSize=None, seed=2, r=1.0, bgtransforms=None, onlyfg_rate=0.5):
        if mode != 'scene':
            assert TenCrop is False, f"TenCrop is set only for scene classification but not {mode}"
        else:
            # for scene and segment
            self.mean = [0.485, 0.456, 0.406]
            self.STD = [0.229, 0.224, 0.225]
        # print(f"Evaluating {split} set")
        self.dbdir = dbdir
        self.split = split
        self.TenCrop = TenCrop
        self.mode = mode
        self.bgtransforms = bgtransforms # used for bgtrain and save_testdata
        self.outputSize = outputSize
        self.onlyfg_rate = onlyfg_rate
        # class foldername, e.g., '00_dog'
        foldernames = glob(os.path.join(dbdir, split, "*"))
        assert len(foldernames) > 0, "No folders found, please check database dir"
        self.nb_classes = len(foldernames)
        self.overlap = overlap
        for i, folder in enumerate(foldernames):
            foldernames[i] = os.path.basename(folder)
        # this is a special case when only considering overlapped images for all the bgdb
        if overlap != '':
            _ext = '.npy' if mode == 'mask' else '.JPEG'
            data = np.load(overlap).astype(object)
            foldernames = sorted(foldernames)
            for i in range(len(data)):
                data[i][1] = int(data[i][1])
                _foldername = foldernames[data[i][1]]
                assert int(_foldername[:2]) == data[i][1], "Labels are not aligned"
                data[i][0] = os.path.join(dbdir, split, _foldername, data[i][0]+_ext)
            self.data = data
        else:
            filenames = []
            for folder in foldernames:
                if split == "all":
                    filenames_tr = glob(os.path.join(dbdir, "train", folder, "*"))
                    filenames_val = glob(os.path.join(dbdir, "val", folder, "*"))
                    nb_imgs = len(filenames_tr) + len(filenames_val)
                    array = np.empty([nb_imgs, 2], dtype=object)
                    array[:len(filenames_tr), 0] = filenames_tr
                    array[len(filenames_tr):, 0] = filenames_val
                else:
                    filenames_split = glob(os.path.join(dbdir, split, folder, "*"))
                    nb_imgs = len(filenames_split)
                    array = np.empty([nb_imgs, 2], dtype=object)
                    array[:, 0] = filenames_split
                array[:, 1] = int(folder[:2])
                filenames.append(array)
            self.data = np.concatenate(filenames)
        if r < 1.0:
            np.random.seed(seed)
            np.random.shuffle(self.data)
            if r < 1.0:
                self.data = self.data[:int(r*len(self.data))]
        else:
            np.sort(self.data, axis=0)
        if mode == 'scene':
            self.get_scene_transform()
        # if mode == 'bgaug':
        #     self.background_db = BG20K(BG20K_DIR,
        #                                TenCrop=False,
        #                                mode='bgtest', 
        #                                split=split, 
        #                                outputSize=224,
        #                                seed=seed, 
        #                                r=r)

    # for mode == 'scene' only
    def get_scene_transform(self):
        if not self.TenCrop:
            self.val_transforms_img = transforms.Compose([
                transforms.CenterCrop(self.outputSize),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.STD)
            ])
        else:
            self.val_transforms_img = transforms.Compose([
                transforms.TenCrop(self.outputSize),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(
                    lambda crops: torch.stack([transforms.Normalize(self.mean, self.STD)(crop) for crop in crops])),
            ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        # image = read_image(img_path)
        # if image.shape[0] == 1:
        #   image = read_image(img_path, ImageReadMode.RGB)
        # image = image.type(torch.FloatTensor)/255
        if self.mode == 'scene':
            image = Image.open(img_path)
            try:
                img = self.val_transforms_img(image)
            except:
                image = TF.resize(image, (256, 256))
                img = self.val_transforms_img(image)
            if not self.TenCrop:
                assert img.shape[0] == 3 and img.shape[1] == self.outputSize and img.shape[2] == self.outputSize
            else:
                assert img.shape[0] == 10 and img.shape[2] == self.outputSize and img.shape[3] == self.outputSize
            image = transforms.ToTensor()(image)
            if self.outputSize is not None:
                image = TF.resize(image, (self.outputSize, self.outputSize))
            # Original: resized ...
            self.sample = {'Image': img, 'Label': label, 'Original': image, 'path': os.path.basename(img_path)}
        elif self.mode == 'segment':
            image = Image.open(img_path)
            # image.convert('RGB')
            batch_data = transforms.ToTensor()(image)
            segSize = torch.tensor([batch_data.shape[1], batch_data.shape[2]])
            # batch_data = torch.unsqueeze(batch_data, 0)
            self.sample = {"img_data": batch_data, "Label": label, "segSize": segSize, "path": os.path.basename(img_path)}
        elif self.mode == 'mask':
            mask = np.load(img_path)
            if mask.shape[0] != self.outputSize or mask.shape[1] != self.outputSize:
                mask = TF.resize(mask, (self.outputSize, self.outputSize))
                import ipdb; ipdb.set_trace()
            self.sample = {"mask": mask, "Label": label}
        elif self.mode == 'bgtest':
            image = Image.open(img_path)
            batch_data = transforms.ToTensor()(image)
            if self.outputSize is not None:
                batch_data = TF.resize(batch_data, (self.outputSize, self.outputSize))
                # batch_data = transforms.CenterCrop(self.outputSize)(batch_data)
            if self.bgtransforms is not None:
                batch_data = self.bgtransforms(batch_data)
            self.sample = {"img_data": batch_data, "Label": label, "path": os.path.basename(img_path)}
        elif self.mode == 'onlyfg':
            # image = Image.open(img_path)
            # mask_name = img_path.replace('original', 'fg_mask')
            # mask_name = os.path.basename(mask_name)[:-len('.JPEG')]+'.npy'
            # if os.path.exists(mask_name):
            #     mask = np.load(mask_name)
            #     if self.mode == 'bgaug':
            #         background = self.background_db[np.random.randint(len(self.background_db))]
            #         background = torch.multiply(background, ~mask)
            #     else:
            #         background = torch.rand([3, 224, 224])
            #     foreground = torch.multiply(image, mask)
            #     image = torch.add(foreground, background)
            onlyfg_name = img_path.replace('original', 'only_fg')
            if os.path.exists(onlyfg_name):
                if np.random.random() > self.onlyfg_rate:
                    image = Image.open(onlyfg_name)
                else:
                    image = Image.open(img_path)
            else:
                image = Image.open(img_path)
                print(f"{onlyfg_name} does not exist, please check the dbs paths")
            batch_data = transforms.ToTensor()(image)
            if self.outputSize is not None:
                batch_data = TF.resize(batch_data, (self.outputSize, self.outputSize))
                # batch_data = transforms.CenterCrop(self.outputSize)(batch_data)
            if self.bgtransforms is not None:
                batch_data = self.bgtransforms(batch_data)
            self.sample = {"img_data": batch_data, "Label": label, "path": os.path.basename(img_path)}
        elif self.mode == 'triplet':
            onlyfg_name = img_path.replace('original', 'only_fg')
            ofg_image = None
            if os.path.exists(onlyfg_name):
                ofg_image = Image.open(onlyfg_name)
                ofg_image = transforms.ToTensor()(ofg_image)            
            ori_image = Image.open(img_path)
            ori_image = transforms.ToTensor()(ori_image)
            if self.outputSize is not None:
                ori_image = TF.resize(ori_image, (self.outputSize, self.outputSize))
                if ofg_image is not None:
                    ofg_image = TF.resize(ofg_image, (self.outputSize, self.outputSize))
            if self.bgtransforms is not None:
                ori_image = self.bgtransforms(ori_image)
                if ofg_image is not None:
                    ofg_image = self.bgtransforms(ofg_image)
            triplet_data = torch.cat((torch.unsqueeze(ofg_image, dim=0),torch.unsqueeze(ori_image, dim=0)), axis=0)
            self.sample = {"img_data": ori_image, "triplet_data": triplet_data, "Label": label, "path": os.path.basename(img_path)}
        else:
            print(f'Unidentified mode: {mode}')
            self.sample = None
        return self.sample


# used for generating background candidates
class BG20K():

    def __init__(self, dbdir, TenCrop, mode, split="testval", outputSize=None, seed=2, r=1.0):
        if mode != 'scene':
            assert TenCrop is False, f"TenCrop is set only for scene classification but not {mode}"
        self.outputSize = outputSize
        self.TenCrop = TenCrop
        self.mode = mode
        self.filenames = glob(os.path.join(dbdir, split, "*"))
        self.mean = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        assert len(self.filenames) > 0, "No folders found, please check database dir"
        if r < 1.0:
            np.random.seed(seed)
            np.random.shuffle(self.filenames)
            if r < 1.0:
                self.filenames = self.filenames[:int(r*len(self.filenames))]
        else:
            np.sort(self.filenames, axis=0)

        if mode == 'scene':
            self.get_scene_transform()

    # for mode == 'scene' only
    def get_scene_transform(self):
        if not self.TenCrop:
            self.val_transforms_img = transforms.Compose([
                transforms.CenterCrop(self.outputSize),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.STD)
            ])
        else:
            self.val_transforms_img = transforms.Compose([
                transforms.TenCrop(self.outputSize),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(
                    lambda crops: torch.stack([transforms.Normalize(self.mean, self.STD)(crop) for crop in crops])),
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        if self.mode == 'scene':
            image = Image.open(img_path)
            try:
                img = self.val_transforms_img(image)
            except:
                image = TF.resize(image, (256, 256))
                img = self.val_transforms_img(image)
            if not self.TenCrop:
                assert img.shape[0] == 3 and img.shape[1] == self.outputSize and img.shape[2] == self.outputSize
            else:
                assert img.shape[0] == 10 and img.shape[2] == self.outputSize and img.shape[3] == self.outputSize
            image = transforms.ToTensor()(image)
            if self.outputSize is not None:
                image = TF.resize(image, (self.outputSize, self.outputSize))
            # Original: resized ...
            # Dummy label
            self.sample = {'Image': img, 'Original': image, "Label": -1, 'path': os.path.basename(img_path)}
        elif self.mode == 'segment':
            image = Image.open(img_path)
            # image.convert('RGB')
            batch_data = transforms.ToTensor()(image)
            segSize = torch.tensor([batch_data.shape[1], batch_data.shape[2]])
            # batch_data = torch.unsqueeze(batch_data, 0)
            # size can be different.. solution: batch size 1?
            # Dummy label
            self.sample = {"img_data": batch_data, "segSize": segSize, "Label": -1, "path": os.path.basename(img_path)}
        elif self.mode == 'bgtest':
            image = Image.open(img_path)
            batch_data = transforms.ToTensor()(image)
            if self.outputSize is not None:
                batch_data = TF.resize(batch_data, (self.outputSize, self.outputSize))
                # batch_data = transforms.CenterCrop(self.outputSize)(batch_data)
            # Dummy label
            self.sample = {"img_data": batch_data, "Label": -1, "path": os.path.basename(img_path)}
        else:
            print(f'Unidentified mode: {mode}')
            self.sample = None
        return self.sample



class LeakyDataset():
    def __init__(self, traindata, testdata, r, seed=2):
        self.r = r
        gen = torch.Generator().manual_seed(seed)
        len_text = len(testdata)
        nb_leak = int(r*len_text)
        testdata, _ = random_split(testdata, [nb_leak, len_text-nb_leak], generator=gen)
        self.data = torch.utils.data.ConcatDataset([traindata, testdata])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]['img_data'], self.data[idx]['Label']
        return {'img_data': image, 'Label': label}
        # image, label = self.data[idx]
        # return (image, label)



class NoisyDataset():
    def __init__(self, data, noise, anomaly, nb_class=9):
        self.data = data
        self.noise = noise
        self.anomaly = anomaly
        self.nb_class = nb_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # image, label = self.data[idx]
        image, label = self.data[idx]['img_data'], self.data[idx]['Label']
        if self.anomaly == "2" and np.random.random() < self.noise:
            image = torch.rand(image.shape)
        if self.anomaly == "7" and np.random.random() < self.noise:
            label = np.random.randint(self.nb_class)
        return {'img_data': image, 'Label': label}
        # return (image, label)
