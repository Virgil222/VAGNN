"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, posItem, negItem, posAuthor, negAuthor, users2, posAuthor2, negAuthor2):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg, users2, pos2, neg2):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        users_emb2 = self.embedding_user(users2.long())
        pos_emb2 = self.embedding_author(pos2.long())
        neg_emb2 = self.embedding_author(neg2.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        pos_scores2 = torch.sum(users_emb2 * pos_emb2, dim=1)
        neg_scores2 = torch.sum(users_emb2 * neg_emb2, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores)) + torch.mean(
            nn.functional.softplus(neg_scores2 - pos_scores2))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users)) + (1 / 2) * (users_emb2.norm(2).pow(2) +
                                                                                       pos_emb2.norm(2).pow(2) +
                                                                                       neg_emb2.norm(2).pow(2)) / float(
            len(users2))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_authors = self.dataset.n_authors
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_author = torch.nn.Embedding(
            num_embeddings=self.num_authors, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            nn.init.normal_(self.embedding_author.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            self.embedding_author.weight.data.copy_(torch.from_numpy(self.config['author_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()



        # print("save_txt")
        self.q = torch.nn.Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim), requires_grad=True)
        self.q.data.fill_(0.25)




        self.ssl_temp=self.config['ssl_temp']
        # nn.init.xavier_normal_(self.q)




    def one_propagate(self, graph, A_feature, B_feature):

        # propagate
        features = torch.cat([A_feature, B_feature])
        all_features = [features]
        for i in range(self.n_layers):
            if isinstance(graph, list):
                all_emb = torch.sparse.mm(graph[i], features)
            else:
                all_emb = torch.sparse.mm(graph, features)

            all_features.append(all_emb)

        all_features = torch.stack(all_features, dim=1)
        light_out = torch.mean(all_features, dim=1)
        A_feature, B_feature = torch.split(light_out, [A_feature.size()[0], B_feature.size()[0]])

        return A_feature, B_feature

    def computer(self, ui_graph, ua_graph):
        """
        propagate methods for lightGCN
        """

        #  =============================  item level propagation  =============================
        atom_users_feature, atom_items_feature = self.one_propagate(
            ui_graph, self.embedding_user.weight, self.embedding_item.weight)
        atom_authors_feature = F.normalize(torch.matmul(self.Graph[2], atom_items_feature))
        # tiktok 最佳 0.5 0.5
        atom_items_feature = 0.5 * F.normalize(
            torch.matmul(self.Graph[4], atom_items_feature)) + 0.5 * atom_items_feature

        #  ============================= author level propagation =============================
        non_atom_users_feature, non_atom_authors_feature = self.one_propagate(
            ua_graph, self.embedding_user.weight, self.embedding_author.weight)
        non_atom_items_feature = F.normalize(torch.matmul(self.Graph[3], non_atom_authors_feature))

        users_feature = [atom_users_feature, non_atom_users_feature]
        authors_feature = [atom_authors_feature, non_atom_authors_feature]
        items_feature = [atom_items_feature, non_atom_items_feature]

        return users_feature, items_feature, authors_feature

    def getUsersRating(self, users):
        all_users, all_items, all_authors = self.computer(self.Graph[0], self.Graph[1])
        # users = [aa.tolist() for aa in users]
        # users_emb1 = all_users[0][users.long()]
        # users_emb2 = all_users[1][users.long()]
        # items_emb1 = all_items[0]
        # items_emb2 = all_items[1]
        users_feature_atom, users_feature_non_atom = [i[users] for i in all_users]  # batch_f
        authors_feature_atom, authors_feature_non_atom = all_authors  # b_f
        items_feature_atom, items_feature_non_atom = all_items  # b_f

        authors_feature_atom = authors_feature_atom[self.dataset.author_list]
        authors_feature_non_atom = authors_feature_non_atom[self.dataset.author_list]

        ui = self.f(torch.mm(users_feature_atom, items_feature_atom.t()) \
                    + torch.mm(users_feature_non_atom, items_feature_non_atom.t()))  # batch_b
        ua = self.f(torch.mm(users_feature_atom, authors_feature_atom.t()) \
                    + torch.mm(users_feature_non_atom, authors_feature_non_atom.t()))

        items_feature = (items_feature_atom + items_feature_non_atom) / 2
        authors_feature = (authors_feature_atom + authors_feature_non_atom) / 2

        a = ui * torch.norm(items_feature, dim=1)  # ui*|i|
        b = ua * torch.norm(authors_feature, dim=1)  # ua*|a|

        c = torch.mm(items_feature, self.q)  # i*q
        d = (torch.sum(c * authors_feature, 1))  # i*q*a

        weight = torch.sigmoid(d)

        # rating = self.f(torch.matmul(users_emb, items_emb.t()))

        return weight * ui + (1 - weight) * ua

        # return ui

    def getEmbedding(self, users, pos_items, neg_items, pos_authors, neg_authors, users2, pos_authors2, neg_authors2,subGraph):
        all_users, all_items, all_authors = self.computer(self.Graph[0], self.Graph[1])
        all_users_sub1, all_items_sub1, all_authors_sub1 = self.computer(subGraph[0], subGraph[2])
        all_users_sub2, all_items_sub2, all_authors_sub2 = self.computer(subGraph[1], subGraph[3])
        #all_users_sub1, all_items_sub1, all_authors_sub1 = self.computer(subGraph[0], self.Graph[1])
        #all_users_sub2, all_items_sub2, all_authors_sub2 = self.computer(subGraph[1], self.Graph[1])
        user_embeddings1 = (all_users_sub1[0] + all_users_sub1[1]) / 2
        user_embeddings2 = (all_users_sub2[0] + all_users_sub2[1]) / 2
        item_embeddings1 = (all_items_sub1[0] + all_items_sub1[1]) / 2
        item_embeddings2 = (all_items_sub2[0] + all_items_sub2[1]) / 2
        author_embeddings1 = (all_authors_sub1[0] + all_authors_sub1[1]) / 2
        author_embeddings2 = (all_authors_sub2[0] + all_authors_sub2[1]) / 2

        # train loader1
        users_emb0 = all_users[0][users]
        users_emb1 = all_users[1][users]
        posItem_emb0 = all_items[0][pos_items]
        posItem_emb1 = all_items[1][pos_items]
        negItem_emb0 = all_items[0][neg_items]
        negItem_emb1 = all_items[1][neg_items]
        posAuthor_emb0 = all_authors[0][pos_authors]
        posAuthor_emb1 = all_authors[1][pos_authors]
        negAuthor_emb0 = all_authors[0][neg_authors]
        negAuthor_emb1 = all_authors[1][neg_authors]

        users_emb_ego = self.embedding_user(users)
        posItem_emb_ego = self.embedding_item(pos_items)
        negItem_emb_ego = self.embedding_item(neg_items)
        posAuthor_emb_ego = self.embedding_author(pos_authors)
        negAuthor_emb_ego = self.embedding_author(neg_authors)

        # train loader2
        users_emb20 = all_users[0][users2]
        users_emb21 = all_users[1][users2]
        posAuthor_emb20 = all_authors[0][pos_authors2]
        posAuthor_emb21 = all_authors[1][pos_authors2]
        negAuthor_emb20 = all_authors[0][neg_authors2]
        negAuthor_emb21 = all_authors[1][neg_authors2]

        users_emb_ego2 = self.embedding_user(users2)
        posAuthor_emb_ego2 = self.embedding_author(pos_authors2)
        negAuthor_emb_ego2 = self.embedding_author(neg_authors2)

        # Normalize embeddings learnt from sub-graph to construct SSL loss
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)
        author_embeddings1 = F.normalize(author_embeddings1, dim=1)
        author_embeddings2 = F.normalize(author_embeddings2, dim=1)

        user_embs1 = F.embedding(users, user_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs1 = F.embedding(pos_items, item_embeddings1)
        item_embs2 = F.embedding(pos_items, item_embeddings2)
        author_embs1 = F.embedding(pos_authors, author_embeddings1)
        author_embs2 = F.embedding(pos_authors, author_embeddings2)

        pos_ratings_user = torch.sum(user_embs1*user_embs2, dim=-1) # [batch_size]
        pos_ratings_item = torch.sum(item_embs1 * item_embs2, dim=-1)  # [batch_size]
        pos_ratings_author = torch.sum(author_embs1 * author_embs2, dim=-1)  # [batch_size]
        tot_ratings_user = torch.matmul(user_embs1,
                                        torch.transpose(user_embeddings2, 0, 1))  # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1,
                                        torch.transpose(item_embeddings2, 0, 1))  # [batch_size, num_items]
        tot_ratings_author = torch.matmul(author_embs1,
                                        torch.transpose(author_embeddings2, 0, 1))  # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]  # [batch_size, num_users]
        ssl_logits_author = tot_ratings_author - pos_ratings_author[:, None]  # [batch_size, num_users]

        return [users_emb0, users_emb1], [posItem_emb0, posItem_emb1], [negItem_emb0, negItem_emb1], \
               [posAuthor_emb0, posAuthor_emb1], [negAuthor_emb0, negAuthor_emb1], \
               users_emb_ego, posItem_emb_ego, negItem_emb_ego, posAuthor_emb_ego, negAuthor_emb_ego, \
               [users_emb20, users_emb21], [posAuthor_emb20, posAuthor_emb21], [negAuthor_emb20, negAuthor_emb21], \
               users_emb_ego2, posAuthor_emb_ego2, negAuthor_emb_ego2, \
               ssl_logits_user,ssl_logits_item,ssl_logits_author

    '''
    def getEmbedding(self, users, pos_items, neg_items, pos_authors, neg_authors,users2, pos_authors2, neg_authors2):
        all_users, all_items, all_authors = self.computer()

        #train loader1
        users_emb0 = all_users[users]
        posItem_emb0 = all_items[pos_items]
        negItem_emb0 = all_items[neg_items]
        posAuthor_emb0 = all_authors[pos_authors]
        negAuthor_emb0 = all_authors[neg_authors]

        users_emb_ego = self.embedding_user(users)
        posItem_emb_ego = self.embedding_item(pos_items)
        negItem_emb_ego = self.embedding_item(neg_items)
        posAuthor_emb_ego = self.embedding_author(pos_authors)
        negAuthor_emb_ego = self.embedding_author(neg_authors)


        #train loader2
        users_emb20 = all_users[users2]
        posAuthor_emb20 = all_authors[pos_authors2]
        negAuthor_emb20 = all_authors[neg_authors2]


        users_emb_ego2 = self.embedding_user(users2)
        posAuthor_emb_ego2 = self.embedding_author(pos_authors2)
        negAuthor_emb_ego2 = self.embedding_author(neg_authors2)

        return users_emb0, posItem_emb0, negItem_emb0,\
               posAuthor_emb0,negAuthor_emb0, \
               users_emb_ego, posItem_emb_ego, negItem_emb_ego, posAuthor_emb_ego, negAuthor_emb_ego,\
               users_emb20, posAuthor_emb20, negAuthor_emb20, \
               users_emb_ego2, posAuthor_emb_ego2, negAuthor_emb_ego2

    '''

    def bpr_loss(self, users, posItem, negItem, posAuthor, negAuthor, users2, posAuthor2, negAuthor2,subGraph):
        (users_emb, posItem_emb, negItem_emb, posAuthor_emb, negAuthor_emb,
         userEmb0, posItemEmb0, negItemEmb0, posAuthorEmb0, negAuthorEmb0,
         users_emb2, posAuthor_emb2, negAuthor_emb2,
         userEmb02, posAuthorEmb02, negAuthorEmb02,
         ssl_logits_user, ssl_logits_item,ssl_logits_author
         ) = self.getEmbedding(users.long(), posItem.long(), negItem.long(), posAuthor.long(), negAuthor.long(),
                               users2.long(), posAuthor2.long(), negAuthor2.long(),subGraph)

        '''

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posItemEmb0.norm(2).pow(2) +
                              negItemEmb0.norm(2).pow(2) +
                              posAuthorEmb0.norm(2).pow(2) +
                              negAuthorEmb0.norm(2).pow(2) +
                              (userEmb02.norm(2).pow(2) +
                               posAuthorEmb02.norm(2).pow(2) +
                               negAuthorEmb02.norm(2).pow(2))
                              ) / float(len(users))
                              
       
        '''
        reg_loss = (1 / 2) * (torch.sum(torch.pow(userEmb0, 2))+
                              torch.sum(torch.pow(posItemEmb0, 2))+
                              torch.sum(torch.pow(negItemEmb0, 2))+
                              torch.sum(torch.pow(posAuthorEmb0, 2))+
                              torch.sum(torch.pow(negAuthorEmb0, 2))+
                              torch.sum(torch.pow(userEmb02, 2))+
                              torch.sum(torch.pow(posAuthorEmb02, 2))+
                              torch.sum(torch.pow(negAuthorEmb02, 2))
                              )/ float(len(users))

        # pos_scores = torch.mul(users_emb, pos_emb)
        # pos_scores = torch.sum(pos_scores, dim=1)
        ui_pos_scores = (torch.sum(users_emb[0] * posItem_emb[0], 1) \
                         + torch.sum(users_emb[0] * posItem_emb[1], 1))
        # neg_scores = torch.mul(users_emb, neg_emb)
        # neg_scores = torch.sum(neg_scores, dim=1)
        ui_neg_scores = (torch.sum(users_emb[0] * negItem_emb[0], 1) \
                         + torch.sum(users_emb[0] * negItem_emb[1], 1))

        # pos_scores2 = torch.mul(users_emb2, pos_emb2)
        # pos_scores2 = torch.sum(pos_scores2, dim=1)
        ua_pos_scores = (torch.sum(users_emb[0] * posAuthor_emb[0], 1) \
                         + torch.sum(users_emb[0] * posAuthor_emb[1], 1))
        # neg_scores2 = torch.mul(users_emb2, neg_emb2)
        # neg_scores2 = torch.sum(neg_scores2, dim=1)
        ua_neg_scores = (torch.sum(users_emb[0] * negAuthor_emb[0], 1) \
                         + torch.sum(users_emb[0] * negAuthor_emb[1], 1))

        ua_pos_scores2 = (torch.sum(users_emb2[0] * posAuthor_emb2[0], 1) \
                          + torch.sum(users_emb2[0] * posAuthor_emb2[1], 1))
        ua_neg_scores2 = (torch.sum(users_emb2[0] * negAuthor_emb2[0], 1) \
                          + torch.sum(users_emb2[0] * negAuthor_emb2[1], 1))

        # pos_weight
        items_feature = (posItem_emb[0] + posItem_emb[1]) / 2
        authors_feature = (posAuthor_emb[0] + posAuthor_emb[1]) / 2
        a = ui_pos_scores * torch.norm(items_feature, dim=1)  # ui*|i|
        b = ua_pos_scores * torch.norm(authors_feature, dim=1)  # ua*|a|

        c = torch.mm(items_feature, self.q)  # i*q
        d = (torch.sum(c * authors_feature, 1))  # i*q*a
        pos_weight = torch.sigmoid(d)

        # neg_weight
        items_feature = (negItem_emb[0] + negItem_emb[1]) / 2
        authors_feature = (negAuthor_emb[0] + negAuthor_emb[1]) / 2
        a = ui_neg_scores * torch.norm(items_feature, dim=1)  # ui*|i|
        b = ua_neg_scores * torch.norm(authors_feature, dim=1)  # ua*|a|

        c = torch.mm(items_feature, self.q)  # i*q
        d = (torch.sum(c * authors_feature, 1))  # i*q*a
        neg_weight = torch.sigmoid(d)

        pos_ui_weight = pos_weight * ui_pos_scores + (1 - pos_weight) * ua_pos_scores
        neg_ui_weight = neg_weight * ui_neg_scores + (1 - neg_weight) * ua_neg_scores

        #bpr_loss = 0.5 * torch.mean(
        #    torch.nn.functional.softplus(neg_ui_weight - pos_ui_weight)) + 0.5 * torch.mean(
        #    torch.nn.functional.softplus(ua_neg_scores2 - ua_pos_scores2))

        # 0.8 0.2 最佳
        bpr_loss = -0.8 * torch.sum(F.logsigmoid(pos_ui_weight - neg_ui_weight)) - 0.2 * torch.sum(
            F.logsigmoid(ua_pos_scores2 - ua_neg_scores2))


        # InfoNCE Loss
        clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
        clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
        clogits_author = torch.logsumexp(ssl_logits_author / self.ssl_temp, dim=1)
        infonce_loss = torch.sum(clogits_user + clogits_item+clogits_author)

        return bpr_loss, reg_loss,infonce_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items, all_authors = self.computer(self.Graph[0], self.Graph[1])
        print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
