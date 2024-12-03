import os
import numpy as np
import pandas as pd
import argparse
import torch
from random import sample
import random
import math
import time
from tqdm import tqdm
from model import CLIP, tokenize
from torch import nn, optim
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
# from multitask_2 import multitask_data_generator
from multitask import multitask_data_generator
from model_g_coop import CoOp
import json
from data_graph import DataHelper
from data_graph_pseudo_label import DataHelperPseudo
from torch.utils.data import DataLoader
import warnings

# Silence all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def sample_from_classes(index_list, class_list, num_samples, seed):
    """
    Samples `num_samples` elements randomly from each class and returns two lists.

    Parameters:
    - index_list (list): List of indices.
    - class_list (list): List of class labels corresponding to the indices.
    - num_samples (int): Number of samples to select from each class.

    Returns:
    - sampled_indices (list): List of sampled indices.
    - sampled_classes (list): List of sampled class labels corresponding to the indices.
    """
    # Combine into a DataFrame
    data = pd.DataFrame({"Index": index_list, "Class": class_list})

    # Sample num_samples from each class
    sampled_data = (
        data.groupby("Class", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), num_samples), random_state=seed))
    )

    # The sampling process is ordered by classes, so shuffle once more so that the data loader does not have to
    sampled_data = sampled_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    sampled_data.to_csv('sample_pseudo.csv', index=False)

    # Convert the sampled data back to lists
    sampled_indices = sampled_data["Index"].tolist()
    sampled_classes = sampled_data["Class"].tolist()

    return sampled_indices, sampled_classes

def filter_by_classes(indices, classes, desired_classes, labels):
    """
    Filters the indices and classes by a set of desired classes.

    Parameters:
    - indices (list): List of indices.
    - classes (list): List of class labels corresponding to the indices.
    - desired_classes (set): Set of classes to filter by.
    - labels (list): string name of each class

    Returns:
    - filtered_indices (list): List of indices corresponding to the desired classes.
    - filtered_classes (list): List of class labels corresponding to the desired classes.
    """
    filtered_indices = [idx for idx, cls in zip(indices, classes) if cls in desired_classes]
    filtered_classes = [labels[cls] for cls in classes if cls in desired_classes]
    return filtered_indices, filtered_classes

def pred_unlabeled_nodes(args, model):
    setup_seed(seed)

    # creating pseudo label using task labels
    task_prompt = []
    for a in range(len(labels)):
        prompt = the_template + labels[a]
        task_prompt.append(prompt)
    # extract all text embeddings of task labels
    all_classes_labels = tokenize(task_prompt, context_length=args.context_length).to(device)
    with torch.no_grad():
        syn_class = model.encode_text(all_classes_labels)

    Data = DataHelper(arr_edge_index, args, unlabeled_ids)
    loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    node_feas = []
    for i_batch, sample_batched in enumerate(loader):
        # idx_train = sample_batched['node_idx'].to(device)
        idx_train = sample_batched['s_n'].to(device)
        with torch.no_grad():
            node_fea = model.encode_image(idx_train, node_f, edge_index)
            node_feas.append(node_fea)

    node_feas = torch.cat(node_feas, dim=0)

    syn_class /= syn_class.norm(dim=-1, keepdim=True)
    node_feas /= node_feas.norm(dim=-1, keepdim=True)
    similarity = (100.0 * node_feas @ syn_class.T).softmax(dim=-1)
    max_logits, max_indices = similarity.max(dim=-1)
    pred = similarity.argmax(dim=-1)
    pred = pred.cpu().numpy().reshape(-1)
    sample_indices, sample_classes = sample_from_classes(unlabeled_ids, pred, args.num_sample, seed)
    return sample_indices, sample_classes


def main(args):
    setup_seed(seed)

    os.makedirs('./res/{}_pseudo'.format(data_name), exist_ok=True)

    clip_model = CLIP(args)
    # clip_model.load_state_dict(torch.load('./res/{}_pseudo/node_ttgt_8&12_0.1.pkl'.format(data_name), map_location=device))
    clip_model.load_state_dict(torch.load('./model/node_ttgt_8&12_0.1.pkl', map_location=device))
    clip_model.to(device)

    pseudo_idx, pseudo_classes = pred_unlabeled_nodes(args, clip_model)

    # task_list, train_idx, val_idx, test_idx = multitask_data_generator(lab_list, labeled_ids, labels, args.k_spt,
    #                                                                    args.k_val, args.k_qry, args.n_way)
    
    checkpoint_org = torch.load('./splits/cora_org_data_seed{}'.format(seed))  
    task_list, train_idx, val_idx, test_idx = checkpoint_org['task_list'], checkpoint_org['train_idx'],\
                                                checkpoint_org['val_idx'], checkpoint_org['test_idx']  
    
    all_acc = []
    f1_list = []
    for j in range(len(task_list)):
        # get only classes in the split
        filtered_idx, filtered_classes = filter_by_classes(pseudo_idx, pseudo_classes, task_list[j], labels)
        # filtered_idx, filtered_classes = filter_by_classes(pseudo_idx, pseudo_classes, labels, labels)

        # train_idx_ts = torch.from_numpy(np.array(train_idx[j])).to(device)
        train_idx_ts = torch.from_numpy(np.array(filtered_idx)).to(device)
        val_idx_ts = torch.from_numpy(np.array(val_idx[j])).to(device)
        test_idx_ts = torch.from_numpy(np.array(test_idx[j])).to(device)

        # train_truth = np.array(lab_list)[np.array(train_idx[j])]
        train_truth = np.array(filtered_classes)
        val_truth = np.array(lab_list)[np.array(val_idx[j])]
        test_truth = np.array(lab_list)[np.array(test_idx[j])]

        task_lables_arr = np.array(labels)[task_list[j]]
        task_labels_dict = dict()
        for i in range(task_lables_arr.shape[0]):
            task_labels_dict[task_lables_arr[i]] = i

        train_truth_ts = [task_labels_dict[train_truth[i]] for i in range(len(train_truth))]
        # train_truth_ts = torch.from_numpy(np.array(train_truth_ts)).to(device)
        train_truth_ts = torch.from_numpy(np.array(train_truth_ts, dtype=np.int64)).to(device)

        val_truth_ts = [task_labels_dict[val_truth[i]] for i in range(len(val_truth))]
        val_truth_ts = torch.from_numpy(np.array(val_truth_ts, dtype=np.int64)).to(device)

        test_truth_ts = [task_labels_dict[test_truth[i]] for i in range(len(test_truth))]
        test_truth_ts = torch.from_numpy(np.array(test_truth_ts)).to(device)

        task_lables = task_lables_arr.tolist()
        # Data = DataHelper(arr_edge_index, args, train_idx[j])
        Data = DataHelperPseudo(arr_edge_index, args, filtered_idx, filtered_classes)
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        # since the classes and nodes are pretty random have to put them into a dictionary
        temp = {}
        for i_batch, sample_batched in enumerate(loader):
            s_n = sample_batched['s_n'].numpy()
            t_n = sample_batched['t_n'].numpy()
            class_ids = sample_batched['class_id']
            for s_n_single, t_n_single, class_id in zip(s_n, t_n, class_ids):
                # convert t_n from numpy to list
                t_n_single = t_n_single.tolist()
                if class_id not in temp:
                    temp[class_id] = []
                temp[class_id] += [s_n_single]
                temp[class_id] += t_n_single

        # each g_text is the prompt for one class
        g_texts = []
        for class_name in task_lables:
            if class_name in filtered_classes:
                g_text = [tit_list[a] for a in temp[class_name]]
                g_texts.append(g_text)

        model = CoOp(args, task_lables, clip_model, g_texts, device)

        best_val = 0
        patience = args.patience
        counter = 0

        task_train_loss = []
        task_train_acc = []
        task_val_loss = []
        task_val_acc = []
        for epoch in tqdm(range(1, args.ft_epoch + 1)):
            # print('----epoch:' + str(epoch))
            model.train()
            train_logits = model.forward(train_idx_ts, node_f, edge_index, train_truth_ts)
            with torch.no_grad():
                train_loss = F.cross_entropy(train_logits, train_truth_ts)
                task_train_loss.append(train_loss.item())
                train_acc = accuracy_score(train_truth_ts.cpu(), train_logits.argmax(dim=1).cpu())
                task_train_acc.append(train_acc)


            model.eval()
            with torch.no_grad():
                res = model.forward(val_idx_ts, node_f, edge_index, val_truth_ts, training=False)
                val_acc = accuracy_score(val_truth_ts.cpu(), res.argmax(dim=1).cpu())
                val_loss = F.cross_entropy(res, val_truth_ts)
                task_val_loss.append(val_loss.item())
                task_val_acc.append(val_acc)
                if val_acc <= best_val:
                    counter += 1
                    if counter >= patience:
                        break
                else:
                    best_val = val_acc
                    torch.save(model, './res/{}_pseudo/g_coop.pkl'.format(data_name))
                    counter = 0
        
        df_loss_dict = {
            'train_loss': task_train_loss,
            'val_loss': task_val_loss
        }

        df_acc_dict = {
            'train_acc': task_train_acc,
            'val_acc': task_val_acc
        }

        # saving loss and accuracy curve for later analysis
        df_loss = pd.DataFrame.from_dict(df_loss_dict)
        df_acc = pd.DataFrame.from_dict(df_acc_dict)
        df_loss.to_csv(f'./loss_acc/task_{j}_loss.csv', index=False)
        df_acc.to_csv(f'./loss_acc/task_{j}_acc.csv', index=False)
        # print('{}th_task_best_val'.format(j), round(best_val, 4))

        best_model = torch.load('./res/{}_pseudo/g_coop.pkl'.format(data_name))
        best_model.eval()
        with torch.no_grad():
            res = model.forward(test_idx_ts, node_f, edge_index, test_truth_ts, training=False)
            test_acc = accuracy_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu())
            all_acc.append(test_acc)
            f1 = f1_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu(), average='macro')
            f1_list.append(f1)

    ans = round(np.mean(all_acc).item(), 4)
    print('acc', ans)

    ans = round(np.mean(f1_list).item(), 4)
    print('macro f1', ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--ft_epoch', type=int, default=50, help='fine-tune epoch')
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)

    parser.add_argument('--edge_coef', type=float, default=0.1)
    parser.add_argument('--neigh_num', type=int, default=3)

    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)

    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--coop_n_ctx', type=int, default=4)
    parser.add_argument('--prompt_lr', type=float, default=0.01)

    parser.add_argument('--position', type=str, default='end')
    parser.add_argument('--class_specific', type=bool, default=False)
    parser.add_argument('--ctx_init', type=bool, default=True)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_sample', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()

    data_name = 'cora'
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)
    # device = torch.device("cpu")
    FType = torch.FloatTensor
    LType = torch.LongTensor

    num_nodes = 0
    tit_list = []
    lab_list = []
    with open('./data/train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            tit_list.append(line[2])
            lab_list.append(line[3])
            num_nodes += 1

    print('num_nodes', num_nodes)

    labeled_ids = []
    unlabeled_ids = []
    for i in range(len(lab_list)):
        if lab_list[i] != 'nan':
            labeled_ids.append(i)
        else:
            unlabeled_ids.append(i)

    print('{} nodes having lables'.format(len(labeled_ids)))

    raw_edge_index = [[], []]
    with open('./data/mapped_edges.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            raw_edge_index[0].append(int(line[0]))
            raw_edge_index[1].append(int(line[1]))

    edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
    arr_edge_index = np.array(edge_index)
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load('./data/node_f.npy')
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(device)

    # label_texts = []
    with open('./data/lab_list.txt', 'r') as f:
        line = f.readline().strip().split('\t')
        label_texts = line

    labels = []
    for i in label_texts:
        if i != 'nan':
            labels.append(i)

    start = time.perf_counter()
    all_acc_list = []
    all_macf1_list = []

    the_list = ['', 'a ', 'an ', 'of ', 'paper of ', 'research of ', 'a paper of ', 'a research of ', 'a model of ',
                'research paper of ', 'a research paper of ']

    seed = args.seed
    print('seed', seed)

    # the_template = the_list[0]
    # change to template = a + [ClassName]
    the_template = the_list[1]

    main(args)
    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
