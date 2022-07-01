import yaml
import torch
import csv
from datasets import BgChallengeDB, BG20K
from nltk.corpus import wordnet as wn
import numpy as np
import scipy
import collections
import imagenet_models
import dill
import h5py
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

def get_scene_classes(class_file_name):
    classes = list()
    with open(class_file_name) as class_file:
        for line in class_file:
            line = line.split()[0]
            split_indices = [i for i, letter in enumerate(line) if letter == '/']
            # Check if there a class with a subclass inside (outdoor, indoor)
            if len(split_indices) > 2:
                line = line[:split_indices[2]] + '-' + line[split_indices[2]+1:]
            classes.append(line[split_indices[1] + 1:])
    return classes


def get_segment_classes(class_file_name):
    names = {}
    with open(class_file_name) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    return names


def get_select_by_class_dict(CONFIG, data):
    class_dict = {}
    for cid in range(CONFIG['BGDB']['num_class']):
        class_dict[cid] = np.where(data[:, 0]!=cid)[0]
    return class_dict


def get_select_by_item_dict(args, processed_data, graph=None):
    item_dict = {}
    if graph is not None:
        for cid in graph:
            for item in graph[cid]['headerTable']:
                if item not in item_dict:
                    feat_idx = args.class_dict[item]
                    item_dict[item] = np.where(processed_data[:, feat_idx]==1)[0]
    else:
        for item in args.scene_class:
            feat_idx = args.class_dict[item]
            item_dict[item] = np.where(processed_data[:, feat_idx]==1)[0]
        for item_idx in args.segment_class:
            item = args.segment_class[item_idx]
            feat_idx = args.class_dict[item]
            item_dict[item] = np.where(processed_data[:, feat_idx]==1)[0]
    return item_dict


def get_classdict(args):
    class_dict = {}
    duplicated_dict = {}
    for sg_idx in range(len(args.segment_class)):
        class_dict[args.segment_class[sg_idx+1]] = sg_idx+1
    for idx, sc_class in enumerate(args.scene_class, sg_idx+2):
        if sc_class not in class_dict:
            class_dict[sc_class] = idx
        else:
            duplicated_dict[sc_class] = [class_dict[sc_class], idx]
    args.class_dict = class_dict
    args.duplicated_dict = duplicated_dict


def get_config(args):
    CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))
    args.scene_class = get_scene_classes(CONFIG['SCENE']['class_file_name']) # scene class
    args.segment_class = get_segment_classes(CONFIG['SEGMENT']['class_file_name']) # segmentation class
    get_classdict(args)
    return CONFIG


def get_dataloder(args, CONFIG, DATASET_DIR, mode, dbname='bgdb', pin_memory=True):
    if mode.lower() == 'scene':
        TEN_CROPS = CONFIG['SCENE']['TEN_CROPS']
        outputSize = 224
    elif mode == 'segment':
        TEN_CROPS = False
        outputSize = None
    elif mode == 'bgtest':
        TEN_CROPS = False
        outputSize = 224
    elif mode == 'mask':
        TEN_CROPS = False
        outputSize = 224
    # Evaluate model on validation set
    if dbname == 'bgdb':
        val_dataset = BgChallengeDB(DATASET_DIR,
                                    overlap=CONFIG['CAND']['overlap'], 
                                    TenCrop=TEN_CROPS,
                                    mode=mode, 
                                    split=CONFIG['CAND']['split'], 
                                    outputSize=outputSize,
                                    seed=CONFIG['CAND']['seed'], 
                                    r=CONFIG['CAND']['ratio'])
    elif dbname == 'bg20k':
        val_dataset = BG20K(DATASET_DIR,
                            TenCrop=TEN_CROPS,
                            mode=mode, 
                            split=CONFIG['CAND']['split'], 
                            outputSize=outputSize,
                            seed=CONFIG['CAND']['seed'], 
                            r=CONFIG['CAND']['ratio'])

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=pin_memory)
    return val_loader


def get_wnwords(anchor_item):
    if '_' in anchor_item:
        words = anchor_item.split('_')
    else:
        words = anchor_item.split('-')
    wn_words = []
    for word in words:
        wn_synsets = wn.synsets(word)
        if len(wn_synsets) > 0:
            wn_words.append(wn_synsets[0])
    if len(wn_words) == 0:
        print(f'cannot find item {anchor_item} in wordNet')

    return wn_words


def compute_wn_similarity(anchor_words, item_words):
    if len(anchor_words) == 0 or len(item_words) == 0:
        return 0
    else:
        scores = []
        for anchor_word in anchor_words:
            for item_word in item_words:
                scores.append(anchor_word.path_similarity(item_word))
        return np.mean(scores)



def preprocess_dataset(args, CONFIG, dataSet):
    _dataSet = dataSet.copy()
    for i, data_point in enumerate(_dataSet):
        segment_selected = (data_point[1:CONFIG['SEGMENT']['num_class']+1] > 0.1) * 1
        data_point[1:CONFIG['SEGMENT']['num_class']+1] = segment_selected

        dataSet[i][1:CONFIG['SEGMENT']['num_class']+1] = dataSet[i][1:CONFIG['SEGMENT']['num_class']+1] / 100
        # softmax for scene
        dataSet[i][CONFIG['SEGMENT']['num_class']+1:] = scipy.special.softmax(data_point[CONFIG['SEGMENT']['num_class']+1:])

        scene_selected = data_point[CONFIG['SEGMENT']['num_class']+1:].argsort()[-CONFIG['CAND']['maxk']:]
        scene_nonselected = data_point[CONFIG['SEGMENT']['num_class']+1:].argsort()[:-CONFIG['CAND']['maxk']]
        data_point[CONFIG['SEGMENT']['num_class']+1:][scene_nonselected] = 0
        data_point[CONFIG['SEGMENT']['num_class']+1:][scene_selected] = 1
        assert np.sum(data_point[1:]) == CONFIG['CAND']['maxk'] + np.sum(segment_selected), "Check dataset preprocessing. Number of selected features not aligned."

        # for duplicated features, if one is present then set both present
        for feat in args.duplicated_dict:
            feat_indices = args.duplicated_dict[feat]
            if data_point[feat_indices[0]] or data_point[feat_indices[1]]:
                data_point[feat_indices[0]] = 1
                data_point[feat_indices[1]] = 1
    return _dataSet



def similarity(ref, cand):
    return np.linalg.norm(ref[1:]-cand[1:])


def load_model(args, CONFIG, net=None, parallel=True):
    device_ids = torch.cuda.device_count()
    print("Number of GPU(s):", device_ids)
    if args.mid == '-1':
        if net is None:
            net = imagenet_models.__dict__['resnet50'](num_classes=CONFIG['BGDB']['num_class'])
        if device_ids == 0:
            checkpoint = torch.load(args.model_path, pickle_module=dill, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(args.model_path, pickle_module=dill)

        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'
        sd = checkpoint[state_dict_path]
        sd = {k[len('module.attacker.model.'):]:v for k,v in sd.items()}
        
        model_dict = net.state_dict()
        # To deal with some compatability issues
        in_sd = {}
        out_sd = {}
        for k in model_dict:
            if k in sd:
                in_sd[k] = sd[k]
            else:
                out_sd[k] = model_dict[k]
        assert len(out_sd) == 0, 'check loading model for the official bgdb resnet50 model'

        # sd = {k: v for k, v in sd.items() if k in model_dict}
        # model_dict.update(sd)
        model_dict.update(in_sd)
        net.load_state_dict(model_dict)

        if device_ids > 0 and parallel:
            net = torch.nn.DataParallel(net)
            net = net.cuda()
        net.eval()
        return net
    else:
        if device_ids == 0:
            try:
                net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
            except:
                # in case training is done using gpu
                state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
                new_state_dict = collections.OrderedDict()
                for k, v in state_dict.items():
                    name = k[len('module.'):] # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                net.load_state_dict(new_state_dict)
        else:
            net = torch.nn.DataParallel(net)
            try:
                net.load_state_dict(torch.load(args.model_path))
            except:
                # in case training is done using gpu
                state_dict = torch.load(args.model_path)
                new_state_dict = collections.OrderedDict()
                for k, v in state_dict.items():
                    name = "module." + k
                    new_state_dict[name] = v
                net.load_state_dict(new_state_dict)
            net = net.cuda()
        net.eval()
        return net




# get_freqItemRank
def preprocess_graph(graph):
    for cid in graph:
        graph[cid]['freqItemRank'] = sorted(graph[cid]['headerTable'].items(), key=lambda x: -x[1])

def load_graph(CONFIG):
    graph = np.load(CONFIG['CAND']['graph_path'], allow_pickle=True).item()
    preprocess_graph(graph)
    return graph

def load_items(path, split):
    with h5py.File(path, "r") as hdf5_file:
        data = hdf5_file[f"{split}/data"][:]
    nb_examples = len(data)
    return data, nb_examples



def get_relations(mat, l2_norm=True, flip_y=True):
    dim = [len(mat), mat[0].shape[0], mat[0].shape[0]]
    corrcoefs, cos_dists, l2_dists = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    for i, m in enumerate(mat):
        corrcoefs[i] = np.corrcoef(m)
        cos_dists[i] = cosine_distances(m)
        l2_dists[i] = euclidean_distances(m)
        if l2_norm:
            nb_examples = m.shape[1]
            l2_dists[i] /= np.sqrt(nb_examples)
    if flip_y:
        corrcoefs = np.flip(corrcoefs, axis=1)
        cos_dists = np.flip(cos_dists, axis=1)
        l2_dists = np.flip(l2_dists, axis=1)
    return corrcoefs, cos_dists, l2_dists


def get_continuity(imgs, flip_y=True):
    def get_diagonal_std(img, i, forward):
    # return the i-th "diagonal" std starting from the i-th row
        values = []
        j = 0
        while i>=0 and i<len(img):
            values.append(img[i, j])
            if forward:
                i -= 1
            else:
                i += 1
            j += 1
        return np.std(values), len(values)
    results = []
    for img in imgs:
        # fillin only when rotation -1, 0, 1 are the same
        # img = fillin_relation_diagonal(img.copy(), flip_y)
        d_std = []
        dim = img.shape[0]
        m = np.sum(img) / (dim*dim-dim)
        for i in range(1, dim):
            std, w = get_diagonal_std(img, i, forward=flip_y)
            # d_std.append(std)
            d_std.append(w*std/dim)
        # results.append(np.mean(d_std))
        results.append(np.sum(d_std)/m)
    return results


def get_asymmetry(imgs, flip_y=True):
    # Symmetric about the second diagonal
    results = []
    for img in imgs:
        dim = img.shape[0]
        m = np.sum(img) / (dim*dim-dim) # exclude the primary diagonal
        values = []
        if flip_y:
            for i in range(dim):
                for j in range(i):
                    values.append(np.abs(img[i, j]-img[j, i]))
        else:
            for i in range(dim):
                for j in range(dim-i):
                    values.append(np.abs(img[i, j]-img[dim-1-j, dim-1-i]))
        results.append(np.mean(values)/m)
    return results


def merge_relations(relations, nb_angles):
    gap = 2
    h, w = nb_angles + 2 * gap, nb_angles * 2 + 3 * gap
    result = np.zeros([len(relations), h, w])
    for j, relation in enumerate(relations):
        for i, r in enumerate(relation):
            result[j][gap:gap+nb_angles, (i+1)*gap+i*nb_angles:(i+1)*(gap+nb_angles)] = r
    return result
