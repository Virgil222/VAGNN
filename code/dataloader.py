"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
from os.path import join
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def n_authors(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def trainDataSize2(self):
        raise NotImplementedError

    @property
    def validDict(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    @property
    def author_list(self):
        raise NotImplementedError

    @property
    def allPos2(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, config=world.config, path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1, "valid": 2}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        self.n_author = 0
        train_file = path + '/train.txt'
        ua_file = path + '/user_author_train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        ua_train_file = path + '/user_author_train2.txt'
        ai_file = path + '/author_item.txt'
        size_file = path + '/data_size.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser,trainAuthor = [], [], [], []
        trainUniqueUsers2, trainAuthor2, trainUser2 = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []

        self.traindataSize = 0
        self.traindataSize2 = 0
        self.validDataSize = 0
        self.testDataSize = 0

        with open(size_file) as f:
            self.n_user, self.m_item, self.n_author = [int(s) for s in f.readline().split('\t')][:3]

        # u-i train
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # u-a
        with open(ua_file) as f:
            self.U_B_pairs= list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indice = np.array(self.U_B_pairs, dtype=np.int32)
        values = np.ones(len(self.U_B_pairs), dtype=np.float32)
        self.UserAuthorNet = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.n_user, self.n_author)).tocsr()
        self.trainAuthor = np.array(indice[:, 1])

        # u-a train
        with open(ua_train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    authors = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers2.append(uid)
                    trainUser2.extend([uid] * len(authors))
                    trainAuthor2.extend(authors)
                    self.traindataSize2 += len(authors)
        self.trainUniqueUsers2 = np.array(trainUniqueUsers2)
        self.trainUser2 = np.array(trainUser2)
        self.trainAuthor2 = np.array(trainAuthor2)

        # valid
        with open(valid_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    validUniqueUsers.append(uid)
                    validUser.extend([uid] * len(items))
                    validItem.extend(items)

        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)

        # test
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)

        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)



        # (authors,items)
        with open(ai_file, 'r') as f:
            self.A_I_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indice = np.array(self.A_I_pairs, dtype=np.int32)
        values = np.ones(len(self.A_I_pairs), dtype=np.float32)
        self.ground_truth_a_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.n_author, self.m_item)).tocsr()

        self.A_I_pairs = sorted(self.A_I_pairs, key=lambda x: x[-1])

        # print(self.B_I_pairs)
        self.i_a_dict = dict(list(map(lambda x: x[::-1], self.A_I_pairs)))
        self._authorList = list(self.i_a_dict.values())



        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # (users,authors), bipartite graph
        self.UserAuthorNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainAuthor)),
                                         shape=(self.n_user, self.n_author))
        self.users2_D = np.array(self.UserAuthorNet.sum(axis=1)).squeeze()
        self.users2_D[self.users2_D == 0.] = 1
        self.authors_D = np.array(self.UserAuthorNet.sum(axis=0)).squeeze()
        self.authors_D[self.authors_D == 0.] = 1.

        # (users,authors), bipartite graph
        self.UserAuthorNet2 = csr_matrix((np.ones(len(self.trainUser2)), (self.trainUser2, self.trainAuthor2)),
                                        shape=(self.n_user, self.n_author))
        self.users2_D = np.array(self.UserAuthorNet2.sum(axis=1)).squeeze()
        self.users2_D[self.users2_D == 0.] = 1
        self.authors_D = np.array(self.UserAuthorNet2.sum(axis=0)).squeeze()
        self.authors_D[self.authors_D == 0.] = 1.

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self._allPos2 = self.getUserPosAuthors(list(range(self.n_user)))
        self.__validDict = self.__build_valid()
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def n_authors(self):
        return self.n_author

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def trainDataSize2(self):
        return self.traindataSize2

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def author_list(self):
        return self._authorList

    @property
    def allPos2(self):
        return self._allPos2

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))



    def getSparseGraph(self):

        try:
            pre_adj_mat = sp.load_npz(self.path + '/ui_s_pre_adj_mat.npz')
            print("successfully loaded...")
            ui_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            ui_graph = ui_graph.coalesce().to(world.device)


            pre_adj_mat = sp.load_npz(self.path + '/pooling1_s_pre_adj_mat.npz')
            print("successfully loaded...")
            pool1_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            pool1_graph = pool1_graph.coalesce().to(world.device)


            pre_adj_mat = sp.load_npz(self.path + '/pooling2_s_pre_adj_mat.npz')
            print("successfully loaded...")
            pool2_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            pool2_graph = pool2_graph.coalesce().to(world.device)

            pre_adj_mat = sp.load_npz(self.path + '/pooling3_s_pre_adj_mat.npz')
            print("successfully loaded...")
            pool3_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            pool3_graph = pool3_graph.coalesce().to(world.device)

            pre_adj_mat = sp.load_npz(self.path + '/ua_s_pre_adj_mat.npz')
            print("successfully loaded...")
            ua_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            ua_graph = ua_graph.coalesce().to(world.device)

            pre_adj_mat = sp.load_npz(self.path + '/ua2_s_pre_adj_mat.npz')
            print("successfully loaded...")
            ua_graph2 = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            ua_graph2 = ua_graph2.coalesce().to(world.device)
        except:
            print("generating adjacency matrix")

            # u-i graph
            adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            sp.save_npz(self.path + '/ui_s_pre_adj_mat.npz', norm_adj)
            ui_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            ui_graph = ui_graph.coalesce().to(world.device)

            # pooling garph1 (a-i)
            bundle_size = self.ground_truth_a_i.sum(axis=1) + 1e-8
            ai_graph = sp.diags(1 / bundle_size.A.ravel()) @ self.ground_truth_a_i
            sp.save_npz(self.path + '/pooling1_s_pre_adj_mat.npz', ai_graph)
            pool1_graph = self._convert_sp_mat_to_sp_tensor(ai_graph)
            pool1_graph = pool1_graph.coalesce().to(world.device)

            #  pooling graph2 (i-a)
            sp.save_npz(self.path + '/pooling2_s_pre_adj_mat.npz', ai_graph.T)
            pool2_graph = self._convert_sp_mat_to_sp_tensor(ai_graph.T)
            pool2_graph = pool2_graph.coalesce().to(world.device)

            # pooling graph3 (i-i)
            ia_norm = sp.diags(
                1 / (np.sqrt((self.ground_truth_a_i.T.multiply(self.ground_truth_a_i.T)).sum(
                    axis=1).A.ravel()) + 1e-8)) @ self.ground_truth_a_i.T
            ii_graph = ia_norm @ ia_norm.T
            sp.save_npz(self.path + '/pooling3_s_pre_adj_mat.npz', ii_graph)
            pool3_graph = self._convert_sp_mat_to_sp_tensor(ii_graph)
            pool3_graph = pool3_graph.coalesce().to(world.device)

            # u-a graph
            adj_mat = sp.dok_matrix((self.n_user + self.n_author, self.n_user + self.n_author), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserAuthorNet.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            sp.save_npz(self.path + '/ua_s_pre_adj_mat.npz', norm_adj)
            ua_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            ua_graph = ua_graph.coalesce().to(world.device)

            # u-a graph2
            adj_mat = sp.dok_matrix((self.n_user + self.n_author, self.n_user + self.n_author), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserAuthorNet2.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            sp.save_npz(self.path + '/ua2_s_pre_adj_mat.npz', norm_adj)
            ua_graph2 = self._convert_sp_mat_to_sp_tensor(norm_adj)
            ua_graph2 = ua_graph2.coalesce().to(world.device)

        self.Graph=[ui_graph,ua_graph,ua_graph2,pool1_graph,pool2_graph,pool3_graph]

        return self.Graph




    def __build_valid(self):
        """
        return:
            dict: {user: [items]}
        """
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]
        return valid_data

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        posAuthors = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
            posAuthors.append(self.UserAuthorNet[user].nonzero()[1])
        return posItems,posAuthors

    def getUserPosAuthors(self, users):
        posAuthors = []
        for user in users:
            posAuthors.append(self.UserAuthorNet2[user].nonzero()[1])
        return posAuthors

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
