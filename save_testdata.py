import numpy as np
import os
from utils import get_config, preprocess_dataset, load_model, load_graph, load_items
from utils import get_select_by_class_dict, get_select_by_item_dict
import argparse
import matplotlib.pyplot as plt
from dilation import get_freq_itemsets
from candidates import *
import torch
import torch.nn as nn
from model import VGGnet, SimpleNet

res_mean = torch.tensor([0.4717, 0.4499, 0.3837])
res_std = torch.tensor([0.2600, 0.2516, 0.2575])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ConfigPath', type=str, default="./config.yaml")
    parser.add_argument('--model_dir', type=str, default="./saved_models/")
    parser.add_argument("--mid", type=str, default="-1") # model id
    parser.add_argument('--arch', type=str, default="resnet50")
    parser.add_argument('--selection_mode', type=str, default="aa") # aa: association anlysis, random, close: closest
    parser.add_argument('--cand_dbname', type=str, default='bg20k') # background database
    parser.add_argument('--matrix_dir', type=str, default='./matrices') # background database
    args = parser.parse_args()
    model_name = 'in9l_resnet50.pt' if args.mid == "-1" else f'{args.mid}.pth' # 'resnet50-19c8e357.pth'
    args.model_path = os.path.join(args.model_dir, model_name)
    # args.model_label = os.path.join(args.model_dir, "model_label.txt")
    return args


def fill_conf(conf_matrix, i, label, predictions, confidence):
    conf_matrix[:, i, 0] = i
    conf_matrix[:, i, 1] = label
    conf_matrix[:, i, 2] = predictions.numpy()
    conf_matrix[:, i, 3] = confidence.numpy()


def fill_conv(conv_matrix, i, label, predictions, activations):
    conv_matrix[:, i, 0] = i
    conv_matrix[:, i, 1] = label
    conv_matrix[:, i, 2] = predictions.numpy()

    flat_mat = activations.reshape(activations.shape[0], -1)
    m = np.mean(flat_mat, axis=1)
    std = np.std(flat_mat, axis=1)
    overall_max = np.max(flat_mat, axis=1)
    overall_min = np.min(flat_mat, axis=1)
    mean_mat = np.mean(activations.reshape(activations.shape[0], activations.shape[1], -1), axis=2)
    max_mat = np.max(activations.reshape(activations.shape[0], activations.shape[1], -1), axis=2)
    mean_max = np.max(mean_mat, axis=1)
    mean_std = np.std(mean_mat, axis=1)
    max_mean = np.mean(max_mat, axis=1)
    max_std = np.std(max_mat, axis=1)
    conv_matrix[:, i, 3:] = np.array([m, std, overall_max, overall_min, mean_max, mean_std, max_mean, max_std]).transpose(1, 0)


def save_testnpy(args, CONFIG, bg=True):
    if args.mid == '-1':
        net = None
    else:
        if 'vgg' in args.arch:
            net = VGGnet(args.arch, CONFIG['BGDB']['num_class'])
    model = load_model(args, CONFIG, net=net)
    softmax_fn = nn.Softmax(dim=1)
    preprocess = transforms.Normalize(res_mean, res_std)
    try:
        cand_idx = np.load(CONFIG['CAND']['CANDIDX_PATH'], allow_pickle=True)
    except:
        cand_idx = None
        print("Computing candidx from scratch")

    graph = load_graph(CONFIG)
    # keys: each class has its own graph
    # Todo: data -- keep h5 files instead of numpy
    bgdb_items, nb_exbgdb = load_items(CONFIG['CAND']['bgdb_items'], CONFIG['CAND']['split'])
    if args.cand_dbname == 'bg20k':
        cand_items, nb_excand = load_items(CONFIG['CAND']['bg20k_items'], CONFIG['CAND']['split'])
    elif args.cand_dbname == 'bgdb':
        cand_items, nb_excand = load_items(CONFIG['CAND']['bgdb_items'], CONFIG['CAND']['split'])
    processed_bgdb = preprocess_dataset(args, CONFIG, bgdb_items)
    processed_cand = preprocess_dataset(args, CONFIG, cand_items)
    feat_dict = initiate_featdict(args, cand_items, processed_cand, nb_excand)

    selection_dict = {}
    selection_dict['by_bgdbclass'] = get_select_by_class_dict(CONFIG, processed_bgdb)
    selection_dict['by_canditem'] = get_select_by_item_dict(args, processed_cand, graph=graph)
    dbs = get_databases(args, CONFIG)

    conf_columns = ["idx", "label", "prediction", "confidence"]
    conv_columns = ["idx", "label", "prediction", "mean", "std", "max", "min", "mean_max", "mean_std", "max_mean", "max_std"]
    conf_matrix = np.zeros([CONFIG['CAND']['nb_subsets'], nb_exbgdb, len(conf_columns)])
    conv_matrix = np.zeros([CONFIG['CAND']['nb_subsets'], nb_exbgdb, len(conv_columns)])

    with torch.no_grad():
        _correct = 0
        for i in range(len(processed_bgdb)):
        # for i in [800, 801, 1500]:
            # verify_dbs(processed_bgdb, i, dbs)
            label = processed_bgdb[i][0]
            feat_dict['datapoint_idx'] = i
            if bg:
                if args.selection_mode == "close":
                    close_cands = get_closestcands(CONFIG, bgdb_items[i], cand_items)
                    cand_params = get_cand_params(close_cands, None, None, None, i)
                    cand_imgs = generate_cands(args, dbs, cand_params)
                elif args.selection_mode == "random":
                    random_cands = get_randomcands(CONFIG, nb_excand)
                    cand_params = get_cand_params(random_cands, None, None, None, i)
                    cand_imgs = generate_cands(args, dbs, cand_params)
                elif args.selection_mode == "aa":
                    if cand_idx is None:
                        feat_dict['o_feat'] = bgdb_items[i]
                        feat_dict['p_feat'] = processed_bgdb[i]
                        freq_itemsets, freq_items, fuzzed_items, cand_dict = get_freq_itemsets(args, CONFIG, graph, feat_dict, selection_dict)
                    else:
                        cand_dict = cand_idx[i]
                    cand_params = get_cand_params(cand_dict, None, None, None, i)
                    cand_imgs = generate_cands(args, dbs, cand_params)
                cand_imgs = preprocess(cand_imgs)
                if torch.cuda.is_available():
                    cand_imgs.cuda()
                # plt.imshow(torchvision.utils.make_grid(cand_imgs).permute(1, 2, 0))
                # plt.show()                
                ins = model.inspect(cand_imgs)
                out = ins["Linear_0"]
                confidence, predictions = torch.max(softmax_fn(out), axis=1)
                _correct += (predictions[0]==label).item()
                fill_conf(conf_matrix, i, label, predictions, confidence)
                fill_conv(conv_matrix, i, label, predictions, ins['Conv-1'].cpu().numpy())
            else:
                img = preprocess(dbs['original'][i]['img_data'])
                out = model(torch.unsqueeze(img, axis=0))
                confidence, predictions = torch.max(softmax_fn(out), axis=1)
                _correct += (predictions==label).item()
            print(f"Finished {i} / {nb_exbgdb} data object")

        if bg:
            matrix_dir = os.path.join(args.matrix_dir, args.mid)
            if not os.path.exists(matrix_dir):
                os.makedirs(matrix_dir)
            conf_filename = f"bgconf_{args.selection_mode}.npy"
            np.save(os.path.join(matrix_dir, conf_filename), conf_matrix)
            conv_filename = f"bgconv_{args.selection_mode}.npy"
            np.save(os.path.join(matrix_dir, conv_filename), conv_matrix)
        print(f"Test Acc: {_correct/len(processed_bgdb)}")



if __name__ == "__main__":
    args = argparser()
    CONFIG = get_config(args)
    # save_testnpy(args, CONFIG, bg=False) # clean data testing
    save_testnpy(args, CONFIG, bg=True) # bg is true when conducting background invariance test
