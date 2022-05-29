import yaml
import argparse
import torch
import os
from utils import get_config, preprocess_dataset, similarity
from utils import get_select_by_class_dict, get_select_by_item_dict
import matplotlib.pyplot as plt
import numpy as np
import h5py
from itertools import chain, combinations
import collections
from datasets import BgChallengeDB
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ConfigPath', type=str, default="./config.yaml")
    args = parser.parse_args()
    return args


def select_freqItems(graph, itemTable, label=None):
    freq_items = {}
    for item in itemTable:
        for cid in graph:
            if cid is not label and item in graph[cid]['headerTable']:
                freq_items[item] = itemTable[item]
                break
    return freq_items


# find_mode: 'first' -- break when a candidate is found, 'all' -- traverse all cid
# visited: itemset
# selected_itemsets: cand_idx
# selected: cand_idx
def itemsetFuzz(params, itemset, score_dict, cands, visited, selected, selected_itemsets, dist_dict, find_mode='first'):

    graph = params['graph']
    label = params['label']

    def rank_item(item):
        _score, _flag = score_dict[item]
        if _flag is None:
            _flag = -1
        return _score + _flag * 10

    def get_order(itemset, items={}):
        scores = np.zeros(len(itemset)-len(items))
        for i, item in enumerate(itemset):
            if item in items:
                continue
            scores[i] = rank_item(item)
        order = np.argsort(scores)
        return order

    # _fuzzkey is an itemset
    # to find the best fuzzed candidate of class 'cid' for the given _fuzzkey
    def find_cand(_fuzzkey, cid, min_score, chosen_cand, fuzzed_itemsets, viewed, visited, choose_mode):
        iffound = False
        for cand in graph[cid]['fuzz_dict'][_fuzzkey]:
            _freq, fuzzed_item = graph[cid]['fuzz_dict'][_fuzzkey][cand]
            if fuzzed_item in fuzzed_dict or cand in selected_itemsets:
                continue
            if cand not in viewed:
                cand_idx, fuzz_flag, cand_score = choose_candidx(params, cand, visited, dist_dict, choose_mode) # cand is an itemset
                viewed[cand] = (cand_idx, fuzz_flag, cand_score)
            else:
                cand_idx, fuzz_flag, cand_score = viewed[cand]
            if fuzz_flag:
                continue
            if cand_score < min_score[cid==label]:
                min_score[cid==label] = cand_score
                fuzzed_itemsets[cid==label][cand] = (fuzzed_items | set([fuzzed_item]), cand_idx)
                chosen_cand[cid==label] = cand
                iffound = True
        return iffound


    # given a new_itemset, fuzzing the items in order
    def fuzz_item(order, new_itemset, viewed, visited, choose_mode):
        for idx in order:
            _fuzzkey = set(new_itemset)
            _fuzzkey.remove(new_itemset[idx])
            # TODO: the selection process can be improved
            fuzzed_itemsets = collections.defaultdict(dict)
            chosen_cand = collections.defaultdict(dict)
            min_score = {True: 999, False: 999} # True cid == label
            for cid in graph:
                if cid == label:
                    continue
                iffound = find_cand(frozenset(_fuzzkey), cid, min_score, chosen_cand, fuzzed_itemsets, viewed, visited, choose_mode)
                # find_mode: 'first' -- break when a candidate is found, 'all' -- traverse all cid
                if find_mode == 'first' and iffound:
                    break
            if len(fuzzed_itemsets[False]) > 0:
                cand = chosen_cand[False]
                return cand, fuzzed_itemsets[False][cand]        
            find_cand(frozenset(_fuzzkey), label, min_score, chosen_cand, fuzzed_itemsets, viewed, visited, choose_mode)
            if len(fuzzed_itemsets[True]) > 0:
                cand = chosen_cand[True]
                return cand, fuzzed_itemsets[True][cand]
        return None, None


    temp_items = []
    fuzzed_items = set()
    order_idx = 0
    freqItem_idx = 0
    temp_item = None
    order = get_order(itemset)
    len_order = len(order)
    fuzzed_dict = {**score_dict}
    replaced_dict = {}
    new_itemset = list(itemset)


    hard_viewed = {}
    easy_viewed = {}
    countdown = 3
    while countdown > 0 and order_idx < len_order:
        cand_itemset, fuzzed_item = fuzz_item(order, new_itemset, hard_viewed, visited, choose_mode='hard')
        if cand_itemset is not None:
            return cand_itemset, fuzzed_item
        cand_itemset, fuzzed_item = fuzz_item(order, new_itemset, easy_viewed, selected, choose_mode='easy')
        if cand_itemset is not None:
            return cand_itemset, fuzzed_item

        if temp_item in fuzzed_items:
            fuzzed_items.remove(temp_item)
        for temp_item in cands:
            if temp_item in fuzzed_dict:
                continue
            new_itemset[order[order_idx]] = temp_item
            fuzzed_items.add(temp_item)
            temp_items.append(temp_item)
            fuzzed_dict[temp_item] = cands[temp_item]
            break
        else:
            # all items in cands have been fuzzed for the given order_idx
            # fuzzed_item = temp_items[0]
            fuzzed_item = graph[label]['freqItemRank'][freqItem_idx][0]
            while fuzzed_item in score_dict and freqItem_idx < len(graph[label]['freqItemRank']):
                freqItem_idx += 1
                fuzzed_item = graph[label]['freqItemRank'][freqItem_idx][0]
            if freqItem_idx >= len(graph[label]['freqItemRank']):
                fuzzed_item = temp_items[0]

            new_itemset[order[order_idx]] = fuzzed_item
            fuzzed_items.add(fuzzed_item)
            temp_items = []
            replaced_dict[fuzzed_item] = cands[fuzzed_item]
            fuzzed_dict = {**score_dict, **replaced_dict}
            order_idx += 1
            countdown -= 1

    print("!!! CANNOT FIND A FUZZED ITEMSET !!!")
    print("!!! START FUZZING FROM TOP ITEMS !!!")
    # import ipdb; ipdb.set_trace()
    starting_item = graph[label]['freqItemRank'][0][0]
    itemset_length = len(itemset)
    fuzz_key = frozenset((starting_item,))
    while len(fuzz_key) < itemset_length:
        fuzz_keys = graph[label]['fuzz_dict'][fuzz_key]
        for fuzz_key in fuzz_keys:
            break
    cand_idx, fuzz_flag, cand_score = choose_candidx(params, fuzz_key, selected, dist_dict, choose_mode='easy')
    return fuzz_key, (set(), cand_idx)




def dialate_items(CONFIG, graph, main_cands, second_cands, freq_items, label=None):

    def graphFuzz(graph, seed, visited, item_scores, label=None):
        for cid in graph:
            if cid is label:
                continue
            for edge in graph[cid]['graph'].edges(seed[0]):
                if edge[0] == seed[0]:
                    item = edge[1]
                else:
                    item = edge[0]
                if item not in visited:
                    if item not in item_scores:
                        item_scores[item] = (graph[cid]['graph'].edges[edge]['confidence'], None)
                    else:
                        item_scores[item] = (item_scores[item][0] + graph[cid]['graph'].edges[edge]['confidence'], None)

    item_scores = {}
    nb_main = len(main_cands)
    for seed in main_cands:
        graphFuzz(graph, seed, freq_items, item_scores, label)
    sorted_items = sorted(item_scores.items(), key=lambda x: -x[1][0])
    main_cands = main_cands + sorted_items[:max(CONFIG['CAND']['nb_items']-nb_main, 0)]
    second_cands = second_cands + sorted_items[max(CONFIG['CAND']['nb_items']-nb_main, 0):]
    mres, sres = {}, {}
    for data in main_cands:
        mres[data[0]] = data[1]
        freq_items[data[0]] = data[1]
    for data in second_cands:
        sres[data[0]] = data[1]
    return mres, sres


# get the indicies of candidates that have all items in the itemset using the two selection dictionaries.
def get_data_cands(params, itemset, selected):
    selection_dict = params['selection_dict']
    label = params['label']
    nb_excand = params['feat_dict']['nb_excand']
    cand_indices = np.array(range(nb_excand))
    for item in itemset:
        cand_indices = np.intersect1d(cand_indices, selection_dict['by_canditem'][item])
    if label is not None and label > -1:
        _cands = np.intersect1d(cand_indices, selection_dict['by_bgdbclass'][label])
        if len(_cands) > 0:
            if not set(_cands).issubset(selected):
                cand_indices = _cands
    return cand_indices


# return the selected candidate index (the nearest)
def choose_candidx(params, itemset, visited, dist_dict, choose_mode):
    datapoint = params['feat_dict']['p_feat']
    cands = params['feat_dict']['processed_cand']
    datapoint_idx = params['feat_dict']['datapoint_idx']
    # datapoint = params['feat_dict']['o_feat']
    # cands = params['feat_dict']['cand_items']
    fuzz_flag = False
    chosen_candidx = None
    data_cands = get_data_cands(params, itemset, visited)
    min_score = 999
    for cand_idx in data_cands:
        if params['feat_dict']['same_db'] and cand_idx == datapoint_idx:
            continue
        if cand_idx in visited:
            continue
        if choose_mode == 'hard':
            visited.add(cand_idx) # strict rule
        # distance dictionary
        if cand_idx not in dist_dict[datapoint_idx]:
            _score = similarity(datapoint, cands[cand_idx])
            dist_dict[datapoint_idx][cand_idx] = _score
        else:
            _score = dist_dict[datapoint_idx][cand_idx]
        if _score < min_score:
            chosen_candidx = cand_idx
            min_score = _score
    if min_score == 999:
        fuzz_flag = True
    return chosen_candidx, fuzz_flag, min_score


def select_freqItemsets(args, CONFIG, params, freq_items):
    graph = params['graph']

    def powerset(iterable):
        # "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)" 2^n
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    def sort_key(itemset):
        length_score = len(itemset)
        segment_score = 0
        scene_score = 0
        for item in itemset:
            # if it's a fuzzed item
            if item not in freq_items:
                continue
            # if it's an item used by the segmentation model
            if freq_items[item][1] is not None:
                if freq_items[item][1]:
                    segment_score += freq_items[item][0]
                else:
                    scene_score += freq_items[item][0]
        return length_score + (segment_score + scene_score) / 2 # length always has a greater influence

    sorted_items = sorted(freq_items.items(), key=lambda x: -x[1][0])
    main_cands, second_cands = sorted_items[:CONFIG['CAND']['nb_items']], sorted_items[CONFIG['CAND']['nb_items']:]
    main_cands, second_cands = dialate_items(CONFIG, graph, main_cands, second_cands, freq_items)
    assert len(main_cands) == CONFIG['CAND']['nb_items']


    visited = set()
    freq_itemsets = []
    all_itemsets = powerset(main_cands)
    all_itemsets = list(all_itemsets)[::-1]
    level_visited_dict = collections.defaultdict(dict)
    pre = set()
    selected, selected_itemsets = set(), set()
    cand_dict = {}
     # distance dictionary --> dist_dict[data_idx][cand_idx] = distrance_score
    dist_dict = collections.defaultdict(dict)
    fuzz_flag = False


    for is_idx, itemset in enumerate(all_itemsets):
        _len = len(itemset)
        itemset = frozenset(itemset)
        if _len < len(pre):
            level_visited_dict[_len] = {**level_visited_dict[_len+1]}
        score_dict = {**main_cands, **level_visited_dict[_len+1]}
        if _len == 0:
            # comment out if empty set should not be included
            # freq_itemsets.append((frozenset({}), 0))
            continue


        # fuzzed_itemsets / original[cid]: (itemset, fuzzed_items)
        fuzzed_itemsets = {}
        original = {}
        #-----------------------------------------------#
        #       Check the itemset is frequent           #
        #-----------------------------------------------#
        for cid in graph:
            if itemset in graph[cid]['freqItems']:
                original[cid] = (itemset, set())
        if len(original) == 0:
            # If not frequent, fuzz the itemset, and obtain a list of frequent fuzzed_itemsets
            fuzz_flag = True
        

        #-----------------------------------------------#
        #       Check the itemset is not a subset       #
        #-----------------------------------------------#
        # If itemset is a subset of one of the previously selected itemset(s), check confidence < 1
        # Otherwise these two itemsets will have the same candidates
        # params['feat_dict']['same_db'] if not same_db, do not have to fuzz again (optional)
        if not fuzz_flag:
            for pre_selected in selected_itemsets:
                if len(pre_selected) <= _len:
                    continue
                if itemset.issubset(pre_selected):
                    for cid in graph:
                        # if cid is label:
                        #     continue
                        _prefreq = graph[cid]['freqItems'].get(pre_selected, 0)
                        _curfreq = graph[cid]['freqItems'].get(itemset, 0)
                        if _curfreq > _prefreq:
                            break
                    else:
                        # if subset and confidence < 1, fuzz the itemset, and obtain a list of frequent fuzzed_itemsets
                        # fuzzed_item(s) (that are never used at _len+1 level) are selected
                        # All fuzzed itemsets are 100% not a subset of any of the previously selected itemset(s)
                        fuzz_flag = True
                        break

        #-----------------------------------------------#
        #    Check the itemset has enough candidates    #
        #-----------------------------------------------#
        if not fuzz_flag:
            cand_idx, fuzz_flag, _ = choose_candidx(params, itemset, selected, dist_dict, 'easy')


        #-----------------------------------------------#
        #                 Fuzzing loop                  #
        #-----------------------------------------------#
        if fuzz_flag:
            itemset, (fuzzed_items, cand_idx) = itemsetFuzz(params, itemset, score_dict, second_cands, visited, selected, selected_itemsets, dist_dict)
            for fuzzed_item in fuzzed_items:
                if fuzzed_item in second_cands:
                    level_visited_dict[_len][fuzzed_item] = second_cands[fuzzed_item]
                else:
                    level_visited_dict[_len][fuzzed_item] = (0, None)
        # index starts with 1
        cand_dict[is_idx+1] = cand_idx
        score = sort_key(itemset)
        freq_itemsets.append((itemset, score))
        selected.add(cand_idx)
        selected_itemsets.add(itemset)
        visited.add(cand_idx)
        pre = itemset
    return sorted(freq_itemsets, key=lambda x: -x[1]), level_visited_dict[0], cand_dict



def plot_stats(args, stats, cid='overall'):
    nb_nums = len(stats)
    means = [np.mean(stat) for num, stat in stats.items()]
    stds = [np.std(stat) for num, stat in stats.items()]
    labels = [str(stat) for stat in stats]
    width = 0.35       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels, means, width, yerr=stds, label='All examples')
    ax.set_ylabel('Number of itemsets')
    ax.set_xlabel('Length of itemsets (Number of items)')
    # ax.set_title('')
    ax.legend()

    plt.show()
    plt.clf()
    plt.close()


def get_freq_itemsets(args, CONFIG, graph, feat_dict, selection_dict):
    datapoint = feat_dict['p_feat']
    original_datapoint = feat_dict['o_feat']
    feat = datapoint[1:]
    label = datapoint[0]
    itemTable = {}
    # scene_total = 0
    for item_idx, item_value in enumerate(feat, 1):
        if item_value > 0:
            item_value = original_datapoint[item_idx]
            if item_idx <= CONFIG['SEGMENT']['num_class']:
                item = args.segment_class[item_idx]
                segment_flag = True
            else:
                item = args.scene_class[item_idx-CONFIG['SEGMENT']['num_class']-1]
                segment_flag = False
                # scene_total += item_value
            # assert item_idx == args.class_dict[item], 'feature and class dictionary are not aligned'
            if item in itemTable:
                # print(f"Repetitive item: {item}")
                # 'sky', 'tower', 'runway', 'mountain', 'skyscraper'
                # if itemTable[item][1] == segment_flag: segment_flag won't be the same
                #     itemTable[item][0] += item_value                    
                if segment_flag:
                    itemTable[item] = (item_value, True)
            else:
                itemTable[item] = (item_value, segment_flag)

    params = {}
    params['feat_dict'] = feat_dict
    params['selection_dict'] = selection_dict
    params['graph'] = graph
    params['label'] = label

    freq_items = select_freqItems(graph, itemTable, label=None)
    freq_itemsets, fuzzed_items, cand_dict = select_freqItemsets(args, CONFIG, params, freq_items)
    return freq_itemsets, freq_items, fuzzed_items, cand_dict


# demo
def statistics(args, CONFIG):
    graph = np.load(CONFIG['CAND']['graph_path'], allow_pickle=True).item()
    # keys: each class has its own graph
    data, nb_examples = load_data(args, CONFIG)
    processed_data = preprocess_dataset(args, CONFIG, data)

    def update_stats(stats, _stats):
        for num in _stats:
            if num not in stats:
                stats[num] = np.zeros(nb_examples)
            stats[num][i] = _stats[num]

    selection_dict = {}
    selection_dict['by_class'] = get_select_by_class_dict(CONFIG, processed_data)
    selection_dict['by_item'] = get_select_by_item_dict(args, processed_data, graph=graph)
    stats = {}
    for i in range(len(processed_data)):
        freq_itemsets, _, _, _ = get_freq_itemsets(args, CONFIG, i, processed_data, data, graph, selection_dict)
        update_stats(stats, _stats)
        print(f"{i}/{nb_examples}")
    plot_stats(args, stats)


def dilation(args, CONFIG):
    graph = np.load(CONFIG['CAND']['graph_path'], allow_pickle=True).item()
    # keys: each class has its own graph
    data, nb_examples = load_data(args, CONFIG)
    processed_data = preprocess_dataset(args, CONFIG, data)
    DATASET_DIR = CONFIG['BGDB']['ORIGINAL_DIR']
    val_dataset = BgChallengeDB(DATASET_DIR,
                                overlap=CONFIG['CAND']['overlap'], 
                                TenCrop=False,
                                mode='segment', 
                                split=CONFIG['CAND']['split'], 
                                outputSize=None,
                                seed=CONFIG['CAND']['seed'], 
                                r=CONFIG['CAND']['ratio'])
    i = 300
    assert processed_data[i][0] == val_dataset[i]['Label'], 'feat/image Labels are not aligned'

    selection_dict = {}
    selection_dict['by_class'] = get_select_by_class_dict(CONFIG, processed_data)
    selection_dict['by_item'] = get_select_by_item_dict(args, processed_data, graph=graph)
    freq_itemsets, _, _, _ = get_freq_itemsets(args, CONFIG, i, processed_data, data, graph, selection_dict)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(val_dataset[i]['img_data'].permute(1,2,0))

    def frozenset_str(itemset):
        res = []
        for item in itemset:
            res.append(item)
        return ",".join(res)

    cur_y = -0.1
    cur_x = 0.5
    for itemset in freq_itemsets:
        pos = (cur_x, cur_y)
        axes[1].text(pos[0], pos[1], frozenset_str(itemset[0]), size=6, rotation=0.,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),), transform=axes[1].transAxes)
        cur_y += 0.05
        cur_x += 0.0
    plt.show()


if __name__ == "__main__":
    args = argparser()
    CONFIG = get_config(args)
    dilation(args, CONFIG)
    # statistics(args, CONFIG)


