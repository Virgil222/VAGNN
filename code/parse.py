'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=1e-3,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--mess_dropout', type=int, default=0.2,
                        help="using the dropout or not")
    parser.add_argument('--testbatch', type=int,default=128,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='tiktok',
                        help="available datasets: [wechat, tiktok, takatak]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10,20,50]",
                        help="@k test list")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=1)
    parser.add_argument('--epochs', type=int,default=200)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')

    parser.add_argument('--aug_type', type=str, default='rw',
                        help='data augmentation type --- ND: Node Dropout; ED: Edge Dropout; RW: Random Walk')
    parser.add_argument('--ssl_reg', type=int, default=0.7, help='')
    parser.add_argument('--ssl_ratio', type=int, default=0.1, help='')
    parser.add_argument('--ssl_temp', type=int, default=0.3, help='')

    parser.add_argument('--gpu_id', type=str, default='0', help='')

    parser.add_argument('--neg', type=int, default=1, help='')

    return parser.parse_args()


