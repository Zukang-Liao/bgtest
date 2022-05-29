import numpy as np
import os
from datasets import BgChallengeDB, BG20K
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.metrics.pairwise import euclidean_distances
import argparse
from utils import get_config, preprocess_dataset, load_graph, load_items
from utils import get_select_by_class_dict, get_select_by_item_dict
from dilation import get_freq_itemsets

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ConfigPath', type=str, default="./config.yaml")
    parser.add_argument('--plot_dir', type=str, default="./plots/candidates")
    parser.add_argument('--selection_mode', type=str, default="aa") # aa: association anlysis, random, close: closest
    parser.add_argument('--save_plot', action='store_true', default=False) # whether to save plots
    parser.add_argument('--cand_dbname', type=str, default='bg20k') # background database
    args = parser.parse_args()
    args.nb_cols = 8
    args.nb_rows = 4
    return args

def plot_cands(args, cand_params, label, dbs, cand_imgs, plotprefix=""):
    cand_dict = cand_params['cand_dict']
    data_idx = cand_params['data_idx']
    # freq_itemsets = cand_params['freq_itemsets']
    # fuzzed_items = cand_params['fuzzed_items']

    # freq_items = np.zeros(len(cand_params['freq_items']))
    # fuzzed_items = np.zeros(len(cand_params['fuzzed_items']))
    # for i, freq_item in enumerate(cand_params['freq_items']):
        # freq_items[i] = args.class_dict[freq_item]
    # for i, fuzzed_item in enumerate(cand_params['fuzzed_items']):
        # fuzzed_items[i] = args.class_dict[fuzzed_item]
    def plot_original_on_top_left(axes, dbs, data_idx):
        axes[0][0].imshow(dbs['original'][data_idx]['img_data'].permute(1,2,0))
        axes[0][0].grid(False)
        axes[0][0].set_yticks([])
        axes[0][0].set_xticks([])


    plot_dir = os.path.join(args.plot_dir, args.cand_dbname, str(data_idx))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(os.path.join(plot_dir, f"{data_idx}_original.png")):        
        plt.imshow(dbs['original'][data_idx]['img_data'].permute(1,2,0))
        if label > -1:
            plt.title(f"{data_idx} label {label}")
        plt.savefig(os.path.join(plot_dir, f"{data_idx}_original.png"))
        plt.clf()
        plt.close()

    fig, axes = plt.subplots(ncols=args.nb_cols, nrows=args.nb_rows)
    plot_original_on_top_left(axes, dbs, data_idx)
    for is_idx in range(1, len(cand_imgs)):
        row_idx = is_idx // args.nb_cols
        col_idx = is_idx % args.nb_cols
        cand = cand_dict[is_idx]
        # label = dbs['original'][cand]['Label']
        # axes[row_idx][col_idx].imshow(dbs['original'][cand]['img_data'].permute(1,2,0))
        axes[row_idx][col_idx].imshow(dbs['background'][cand]['img_data'].permute(1,2,0))
        axes[row_idx][col_idx].grid(False)
        # axes[row_idx][col_idx].set_title(f'{label}')
        axes[row_idx][col_idx].set_yticks([])
        axes[row_idx][col_idx].set_xticks([])
    axes[-1][-1].set_xticks([])
    axes[-1][-1].set_yticks([])
    # plt.title(f"{data_idx} label {label}")
    plt.savefig(os.path.join(plot_dir, f"{plotprefix}{data_idx}_candidates.png"))
    plt.clf()

    fig, axes = plt.subplots(ncols=args.nb_cols, nrows=args.nb_rows)
    plot_original_on_top_left(axes, dbs, data_idx)
    for is_idx in range(1, len(cand_imgs)):
        row_idx = is_idx // args.nb_cols
        col_idx = is_idx % args.nb_cols
        # label = dbs['original'][cand]['Label']
        axes[row_idx][col_idx].imshow(cand_imgs[is_idx].permute(1,2,0))
        axes[row_idx][col_idx].grid(False)
        # axes[row_idx][col_idx].set_title(f'{label}')
        axes[row_idx][col_idx].set_yticks([])
        axes[row_idx][col_idx].set_xticks([])
    axes[-1][-1].set_xticks([])
    axes[-1][-1].set_yticks([])
    # plt.title(f"{data_idx} label {label}")
    plt.savefig(os.path.join(plot_dir, f"{plotprefix}{data_idx}_candimgs.png"))
    plt.clf()


def generate_cands(args, dbs, cand_params):
    cand_dict = cand_params['cand_dict']
    data_idx = cand_params['data_idx']

    mask = dbs['mask'][data_idx]['mask']
    mask=transforms.ToTensor()(mask)
    original = dbs['original'][data_idx]['img_data']
    foreground = torch.multiply(original, mask)

    cand_imgs = torch.zeros([len(cand_dict)+1, 3, 224, 224])
    # cand_dict[0] = data_idx # only when the same db
    for is_idx in cand_dict:
        cand  = cand_dict[is_idx]
        background = dbs['background'][cand]['img_data']
        background = torch.multiply(background, ~mask)
        new_img = torch.add(foreground, background)
        cand_imgs[is_idx] = new_img
        # plt.imshow(new_img.permute(1,2,0))
    cand_imgs[0] = dbs['original'][data_idx]['img_data']
    return cand_imgs

# def get_candsimilarity(data):
#     return euclidean_distances(data, data)

def get_closestcands(CONFIG, feat, cand_feats):
    # cands = np.argsort(similarity[index])[1: CONFIG['CAND']['nb_subsets']]
    # distances = euclidean_distances(cand_feats[:, 1:], feat[1:])

    distances = np.linalg.norm(cand_feats[:, 1:]-feat[1:], axis=1)
    cands = np.argsort(distances)[:CONFIG['CAND']['nb_subsets']]
    if distances[cands[0]] > 0.01:
        cands = cands[:CONFIG['CAND']['nb_subsets']-1]
    else:
        cands = cands[1:]
    cand_dict = {}
    for i in range(len(cands)):
        cand_dict[i+1] = cands[i]
    return cand_dict

def get_randomcands(CONFIG, nb_examples):
    cands = np.random.choice(range(nb_examples), CONFIG['CAND']['nb_subsets']-1, replace=False)
    cand_dict = {}
    for i in range(len(cands)):
        cand_dict[i+1] = cands[i]
    return cand_dict


def get_databases(args, CONFIG):

    # def plot3(dbs, idx):
    #     fig, axes = plt.subplots(ncols=3, nrows=1)
    #     axes[0].imshow(dbs['original'][idx]['img_data'].permute(1,2,0))
    #     axes[1].imshow(dbs['background'][idx]['img_data'].permute(1,2,0))
    #     axes[2].imshow(dbs['mask'][idx]['mask'])
    #     plt.show()

    original_db = BgChallengeDB(CONFIG['BGDB']['ORIGINAL_DIR'],
                                overlap=CONFIG['CAND']['overlap'], 
                                TenCrop=False,
                                mode='bgtest', 
                                split=CONFIG['CAND']['split'], 
                                outputSize=224,
                                seed=CONFIG['CAND']['seed'], 
                                r=CONFIG['CAND']['ratio'])
    if args.cand_dbname == 'bgdb':
        background_db = BgChallengeDB(CONFIG['BGDB']['BG_T_DIR'],
                                    overlap=CONFIG['CAND']['overlap'], 
                                    TenCrop=False,
                                    mode='bgtest', 
                                    split=CONFIG['CAND']['split'], 
                                    outputSize=224,
                                    seed=CONFIG['CAND']['seed'], 
                                    r=CONFIG['CAND']['ratio'])
    elif args.cand_dbname == 'bg20k':
        background_db = BG20K(CONFIG['BGDB']['BG20K_DIR'],
                              TenCrop=False,
                              mode='bgtest', 
                              split=CONFIG['CAND']['split'], 
                              outputSize=224,
                              seed=CONFIG['CAND']['seed'], 
                              r=CONFIG['CAND']['ratio'])
        # Add your own background candidate database here
    mask_db = BgChallengeDB(CONFIG['BGDB']['MASK_DIR'],
                                overlap=CONFIG['CAND']['overlap'], 
                                TenCrop=False,
                                mode='mask', 
                                split=CONFIG['CAND']['split'], 
                                outputSize=224,
                                seed=CONFIG['CAND']['seed'], 
                                r=CONFIG['CAND']['ratio'])
    dbs = {"original": original_db, "background": background_db, "mask": mask_db}
    # import ipdb; ipdb.set_trace()
    # plot3(dbs, 0)
    return dbs


def get_cand_params(cand_dict, freq_itemsets, fuzzed_items, freq_items, i):
    cand_params = {}
    cand_params['cand_dict'] = cand_dict
    cand_params['freq_itemsets'] = freq_itemsets
    cand_params['fuzzed_items'] = fuzzed_items
    cand_params['freq_items'] = freq_items
    cand_params['data_idx'] = i
    return cand_params


def verify_dbs(processed_data, i, dbs):
    print(f"Processing: {i}")
    label = processed_data[i][0]
    assert processed_data[i][0] == dbs['original'][i]['Label'], 'feat/image Labels are not aligned'
    assert processed_data[i][0] == dbs['background'][i]['Label'], 'feat/image Labels are not aligned'
    assert processed_data[i][0] == dbs['mask'][i]['Label'], 'feat/image Labels are not aligned'


def initiate_featdict(args, cand_items, processed_cand, nb_excand):
    feat_dict = {}
    feat_dict['cand_items'] = cand_items
    feat_dict['processed_cand'] = processed_cand
    feat_dict['nb_excand'] = nb_excand
    if args.cand_dbname == 'bgdb':
        feat_dict['same_db'] = True # if the background database is the same as the bgdb
    else:
        feat_dict['same_db'] = False # if the background database is the same as the bgdb
    return feat_dict


def save_candplots(args, CONFIG):
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
    results = np.empty(len(processed_bgdb), dtype=object)

    for i in range(len(processed_bgdb)):
    # for i in [0, 200, 800, 1200, 1500]:
        # verify_dbs(processed_bgdb, i, dbs)
        label = processed_bgdb[i][0]
        feat_dict['datapoint_idx'] = i

        if args.selection_mode == "close":
            close_cands = get_closestcands(CONFIG, bgdb_items[i], cand_items)
            cand_params = get_cand_params(close_cands, None, None, None, i)
            cand_imgs = generate_cands(args, dbs, cand_params)
            if args.save_plot:
                plot_cands(args, cand_params, label, dbs, cand_imgs, plotprefix="close")
        elif args.selection_mode == "random":
            random_cands = get_randomcands(CONFIG, nb_excand)
            cand_params = get_cand_params(random_cands, None, None, None, i)
            cand_imgs = generate_cands(args, dbs, cand_params)
            if args.save_plot:
                plot_cands(args, cand_params, label, dbs, cand_imgs, plotprefix="random")
        elif args.selection_mode == "aa":
            feat_dict['o_feat'] = bgdb_items[i]
            feat_dict['p_feat'] = processed_bgdb[i]
            freq_itemsets, freq_items, fuzzed_items, cand_dict = get_freq_itemsets(args, CONFIG, graph, feat_dict, selection_dict)
            if args.save_plot:
                cand_params = get_cand_params(cand_dict, freq_itemsets, fuzzed_items, freq_items, i)
                cand_imgs = generate_cands(args, dbs, cand_params)
                plot_cands(args, cand_params, label, dbs, cand_imgs)
            results[i] = cand_dict
        print(f"Finished {i}-th data object")
    np.save(CONFIG['CAND']['CANDIDX_PATH'], results)


if __name__ == "__main__":
    args = argparser()
    CONFIG = get_config(args)
    save_candplots(args, CONFIG)
