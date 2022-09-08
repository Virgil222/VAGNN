'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
from itertools import cycle
from torch.utils.data import DataLoader
CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S, S2 = utils.UniformSample_original(dataset,neg_k)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    posAuthors = torch.Tensor(S[:, 3]).long()
    negAuthors = torch.Tensor(S[:, 4]).long()

    users2 = torch.Tensor(S2[:, 0]).long()
    posAuthors2 = torch.Tensor(S2[:, 1]).long()
    negAuthors2 = torch.Tensor(S2[:, 2]).long()

    users, posItems, negItems,posAuthors,negAuthors = utils.shuffle(users, posItems, negItems,posAuthors,negAuthors)
    users2, posAuthors2, negAuthors2 = utils.shuffle(users2, posAuthors2, negAuthors2)
    #train_loader = DataLoader(S, world.config['bpr_batch_size'], True, num_workers=8, pin_memory=True, drop_last=True)
    #train_loader2 = DataLoader(S2, world.config['bpr_batch_size'], True, num_workers=8, pin_memory=True, drop_last=True)

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.

    train_loader=utils.minibatch(users,posItems,negItems,posAuthors,negAuthors,batch_size=world.config['bpr_batch_size'])
    train_loader2=utils.minibatch(users2,posAuthors2,negAuthors2,batch_size=world.config['bpr_batch_size'])
    #for batch_i,data in enumerate(utils.minibatch(users,posItems,negItems,batch_size=world.config['bpr_batch_size'])):
    subGraph = dataset.getSubGraph()
    for batch_i, data in enumerate(zip(train_loader, cycle(train_loader2))):
        batch_users, batch_posItem, batch_negItem,batch_posAuthor,batch_negAuthor = data[0][0], data[0][1], data[0][2], data[0][3], data[0][4]
        batch_users2, batch_posAuthor2,batch_negAuthor2 = data[1][0], data[1][1], data[1][2]

        batch_users = batch_users.to(world.device)
        batch_posItem = batch_posItem.to(world.device)
        batch_negItem = batch_negItem.to(world.device)
        batch_posAuthor = batch_posAuthor.to(world.device)
        batch_negAuthor = batch_negAuthor.to(world.device)

        batch_users2 = batch_users2.to(world.device)
        batch_posAuthor2 = batch_posAuthor2.to(world.device)
        batch_negAuthor2 = batch_negAuthor2.to(world.device)

        cri = bpr.stageOne(batch_users, batch_posItem, batch_negItem, batch_posAuthor, batch_negAuthor, batch_users2,batch_posAuthor2,batch_negAuthor2,subGraph)
        aver_loss += cri

    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, str, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    if str == 'valid':
        testDict: dict = dataset.validDict
    else:
        testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos[0]):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)

        if multicore == 1:
            pool.close()
        print(results)
        return results
