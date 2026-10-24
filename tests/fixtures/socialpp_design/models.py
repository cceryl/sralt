import torch
import torch.nn as nn
import numpy as np
from layers import *
import scipy.sparse as sp
import torch.nn.functional as F
import random
from utils import normalize_sp
import copy
class MyModel(nn.Module):
    
    def __init__(self):
        super(MyModel, self).__init__()

    def full_query(self, users, train_items=None, method=0):
        # users: [B]
        # 根据train_items修改self.A
        users_emb, items_emb = self.val_forward(users.long(), self.A, method) # 得到users对应的emb和所有item的emb
        # items_emb: [M,H] M is the num of items
        # users_emb: [B,H]
        scores = torch.matmul(users_emb, items_emb.t()) # [B,H]*[M,H].t()=[B,M]
        return scores
    
    def pp_kl(self, mu_p, sig_p, mu_q, sig_q):
        kl_loss = torch.log(sig_q/sig_p)/2 + (sig_p + (mu_p - mu_q)**2)/(2 * sig_q)
        return kl_loss

    def pp_loss(self, pos_u1, pos_u2, neg_u1, neg_u2, defender, def_loss=None):
        pos_u1emb, pos_u2emb = self.gen_uemb(pos_u1), self.gen_uemb(pos_u2)
        pos_uemb = torch.cat([pos_u1emb, pos_u2emb], dim=-1)
        # print(pos_uemb.shape)
        pos_logits = defender(pos_uemb)
        pos_mu, pos_sigma = pos_logits.mean(), pos_logits.var()

        neg_u1emb, neg_u2emb = self.gen_uemb(neg_u1), self.gen_uemb(neg_u2)
        neg_uemb = torch.cat([neg_u1emb, neg_u2emb], dim=-1)
        neg_logits = defender(neg_uemb)
        neg_mu, neg_sigma = neg_logits.mean(), neg_logits.var()
        # print(pos_mu, pos_sigma)
        pp_l1 = torch.sqrt((pos_mu-neg_mu)**2 + (pos_sigma-neg_sigma)**2) # 欧氏距离; 距离越小越好，所以loss不加负号;
        # pp_l2 = 0.5 * self.pp_kl(pos_mu, pos_sigma, neg_mu, neg_sigma) + 0.5 * self.pp_kl(neg_mu, neg_sigma, pos_mu, pos_sigma)
        # all_pploss = pp_l2
        all_pploss = pp_l1
        return all_pploss
    
    def pp_er_loss(self, user, last_uembs=None):
        if last_uembs is None:
            return 0
        all_user_embs = self.gen_all_uemb()['user_embs.weight'] # twinGCN
        this_emb = all_user_embs[user.long()]
        last_emb = last_uembs[user.long()]

        er_loss = torch.abs(this_emb - last_emb).mean()
        return er_loss
       

class DESIGN(MyModel):
    
    def __init__(self, config, args, device):
        super(MyModel, self).__init__()
        self.device = device
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.n_items = config['n_items']
        
        # Graph
        S = config['S']
        self.S = self.convert_numpy_to_tensor(S)
        A = config['A_tr']
        self.A = self.sparse_mx_to_torch_sparse_tensor(A)
        self.RRT = config['RRT_tr']
        self.RRTdrop = None
        
        # training hyper-parameter
        self.hidden = args.hidden
        self.neg = args.neg
        self.std = args.std 
        self.hop = args.hop
        self.drop = args.dropout
        self.decay = args.decay
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.recon_reg = args.recon_reg
        self.recon_drop = args.recon_drop
        self.kl_reg = args.kl_reg
        
        self.sigmoid = torch.nn.Sigmoid()
        
        # layer
        self.user_embs = nn.Embedding(self.n_users, self.hidden) 
        self.item_embs = nn.Embedding(self.n_items, self.hidden)
        nn.init.xavier_uniform_(self.user_embs.weight)
        nn.init.xavier_uniform_(self.item_embs.weight)
        
        self.user1_embs = nn.Embedding(self.n_users, self.hidden)
        self.item1_embs = nn.Embedding(self.n_items, self.hidden)
        self.user2_embs = nn.Embedding(self.n_users, self.hidden)
        self.item2_embs = nn.Embedding(self.n_items, self.hidden)
        nn.init.xavier_uniform_(self.user1_embs.weight)
        nn.init.xavier_uniform_(self.item1_embs.weight)
        nn.init.xavier_uniform_(self.user2_embs.weight)
        nn.init.xavier_uniform_(self.item2_embs.weight)
        
        # xavier_uniform nn.init.xavier_uniform_
        self.socialencoder = SocialGCN(hop=self.hop)
        self.uiencoder = InteractionGCN(hop=self.hop)
        
    
    def convert_numpy_to_tensor(self, adj):
        adj = torch.FloatTensor(adj).to(self.device)
        return adj

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        Graph = torch.sparse.FloatTensor(indices, values, shape)
        Graph = Graph.coalesce().to(self.device)
        return Graph
    
    def edge_dropout(self, sp_adj):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - self.recon_drop)))
        user1_np = np.array(row_idx)[keep_idx]
        user2_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user1_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user1_np, user2_np)), shape=(self.n_users, self.n_users))
        return dropped_adj

    def forward(self, users, pos, neg, epoch=0):
        """
        users:[B] pos:[B] neg:[B, neg] neg=1
        """
        all_user_embs_S = self.socialencoder(self.user_embs.weight, self.S) 
        all_user_embs_A, all_item_embs = self.uiencoder(self.user_embs.weight, self.item_embs.weight, self.A) 
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN

        all_user_embs_social = self.socialencoder(self.user1_embs.weight, self.S) # socialGCN
        all_item_embs_social = self.item1_embs.weight
        
        all_user_embs_rating, all_item_embs_rating = self.uiencoder(self.user2_embs.weight, self.item2_embs.weight, self.A) # ratingGCN
        
        users_emb = all_user_embs[users]
        pos_emb = all_item_embs[pos]
        neg_emb = all_item_embs[neg] # 经过print size 证实是[B,neg,H]
        # x[mask] 当mask为LongTensor https://blog.csdn.net/goodxin_ie/article/details/89672700 就跟nn.Embedding(input) 类似
        
        users_emb_social = all_user_embs_social[users]
        pos_emb_social = all_item_embs_social[pos]
        neg_emb_social = all_item_embs_social[neg]

        users_emb_rating = all_user_embs_rating[users]
        pos_emb_rating = all_item_embs_rating[pos]
        neg_emb_rating = all_item_embs_rating[neg]
        
        return users_emb, pos_emb, neg_emb, all_user_embs_S, all_user_embs_A, users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating
    
    def l2_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.mean(torch.pow(w, 2))

        return 0.5*loss

    def rec_loss(self, user_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(user_emb*pos_emb, dim=1)  # [B]
        neg_scores = torch.sum(user_emb*neg_emb, dim=1)  # [B]
        sup_logits = pos_scores-neg_scores
        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))
        return bpr_loss

    def compute_distill_loss(self, pre_a, pre_b):
        pre_a = self.sigmoid(pre_a)
        pre_b = self.sigmoid(pre_b)
        distill_loss = - torch.mean(pre_b * torch.log(pre_a) + (1 - pre_b) * torch.log(1 - pre_a))
        return distill_loss   
    
    def KL_loss(self, users_emb, pos_emb, neg_emb, users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating):

        pos_emb = pos_emb.unsqueeze(1) # [B,1,H]
        neg_emb = neg_emb.unsqueeze(1) # [B,1,H]
        all_item_embs = torch.cat([pos_emb, neg_emb], dim=1) # [B,2,H] cat方法按输入顺序拼接:先pos_emb, 再negs_emb
        users_emb = users_emb.unsqueeze(1) # [B,1,H]
        pre = torch.mul(users_emb, all_item_embs) # [B,1,H]*[B,2,H]=[B,2,H]
        pre = torch.mean(pre, dim=-1) # [B,2]
        
        pos_emb_social = pos_emb_social.unsqueeze(1)
        neg_emb_social = neg_emb_social.unsqueeze(1)
        all_item_embs_social = torch.cat([pos_emb_social, neg_emb_social], dim=1) 
        users_emb_social = users_emb_social.unsqueeze(1)
        pre_social = torch.mul(users_emb_social, all_item_embs_social) # [B,1,H]*[B,2,H]=[B,2,H]
        pre_social = torch.mean(pre_social, dim=-1) # [B,2]
        
        pos_emb_rating = pos_emb_rating.unsqueeze(1)
        neg_emb_rating = neg_emb_rating.unsqueeze(1)
        all_item_embs_rating = torch.cat([pos_emb_rating, neg_emb_rating], dim=1) 
        users_emb_rating = users_emb_rating.unsqueeze(1)
        pre_rating = torch.mul(users_emb_rating, all_item_embs_rating) # [B,1,H]*[B,2,H]=[B,2,H]
        pre_rating = torch.mean(pre_rating, dim=-1) # [B,2]

        kl_loss = 0
        kl_loss += self.compute_distill_loss(pre, pre_social)
        kl_loss += self.compute_distill_loss(pre, pre_rating)
        kl_loss += self.compute_distill_loss(pre_social, pre)
        kl_loss += self.compute_distill_loss(pre_social, pre_rating)
        kl_loss += self.compute_distill_loss(pre_rating, pre)
        kl_loss += self.compute_distill_loss(pre_rating, pre_social)
        return kl_loss

    def calculate_loss(self, users, pos, neg, epoch):
        """
        Only this function appears in train()
        """
        (users_emb, pos_emb, neg_emb, all_user_embs_S, all_user_embs_A, \
            users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating) = self.forward(users.long(), pos.long(), neg.long(), epoch) # neg_emb:[B, d]
        # print(users_emb, pos_emb, neg_emb, all_user_embs_S, all_user_embs_A, \
        #     users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating)
        bpr_loss = self.rec_loss(users_emb, pos_emb, neg_emb)
        # print(bpr_loss)
        bpr_loss_social = self.rec_loss(users_emb_social, pos_emb_social, neg_emb_social)
        # print(bpr_loss_social)
        bpr_loss_rating = self.rec_loss(users_emb_rating, pos_emb_rating, neg_emb_rating)
        # print(bpr_loss_rating)
        
        reg_loss = self.l2_loss(
            self.user_embs(users.long()),
            self.item_embs(pos.long()),
            self.item_embs(neg.long()),
        )
        reg_loss_social = self.l2_loss(
            self.user1_embs(users.long()),
            self.item1_embs(pos.long()),
            self.item1_embs(neg.long()),
        )
        reg_loss_rating = self.l2_loss(
            self.user2_embs(users.long()),
            self.item2_embs(pos.long()),
            self.item2_embs(neg.long()),
        )

        kl_loss = self.KL_loss(users_emb, pos_emb, neg_emb, users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating)
        # print(kl_loss)
        loss = bpr_loss + bpr_loss_social+ bpr_loss_rating+self.decay*(reg_loss+ reg_loss_social+ reg_loss_rating) + self.kl_reg*kl_loss
        # print('loss', loss)
        return loss
    
    def gen_uemb(self, users):
        all_user_embs_S = self.socialencoder(self.user_embs.weight, self.S) 
        all_user_embs_A, all_item_embs = self.uiencoder(self.user_embs.weight, self.item_embs.weight, self.A) 
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN

        return all_user_embs[users]
    
    def val_forward(self, users, G, mtd=0):
        """
        users:[B] pos:[B] neg:[B, neg] neg=1
        G 有可能修改过了
        """
        if mtd == 0: # UI+social
            all_user_embs_S = self.socialencoder(self.user_embs.weight, self.S) 
            all_user_embs_A, all_item_embs = self.uiencoder(self.user_embs.weight, self.item_embs.weight, G) 
            all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN
        elif mtd == 1: # only social
            # all_user_embs = self.socialencoder(self.user1_embs.weight, self.S) # socialGCN
            # all_item_embs = self.item1_embs.weight
            # all_user_embs = self.socialencoder(self.user_embs.weight, self.S) 
            # all_item_embs = self.item_embs.weight
            
            all_user_embs_S = self.socialencoder(self.user_embs.weight, self.S) 
            _, all_item_embs = self.uiencoder(self.user_embs.weight, self.item_embs.weight, G) 
            all_user_embs = all_user_embs_S
        else:
            # all_user_embs, all_item_embs = self.uiencoder(self.user2_embs.weight, self.item2_embs.weight, G)
            all_user_embs, all_item_embs = self.uiencoder(self.user_embs.weight, self.item_embs.weight, G) 
            # all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN

        return all_user_embs[users], all_item_embs

    def gen_all_uemb(self):
        with torch.no_grad():
            all_user_embs_S = self.socialencoder(self.user_embs.weight, self.S) 
            all_user_embs_A, all_item_embs = self.uiencoder(self.user_embs.weight, self.item_embs.weight, self.A) 
            all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN
            return {'user_embs.weight': all_user_embs}

    def batch_full_sort_predict(self, users, pos, neg_items, epoch):
        (user_emb, pos_emb, negs_emb, _, _, _, _, _, _, _, _) = self.forward(users.long(), pos.long(), neg_items.long(), epoch)
        pos_emb = pos_emb.unsqueeze(1) # [B,1,H]
        all_item_embs = torch.cat([pos_emb, negs_emb], dim=1) # [B,N=1+999,H] cat方法按输入顺序拼接:先pos_emb, 再negs_emb
        user_emb = user_emb.unsqueeze(1) # [B,1,H]
        scores = torch.mul(user_emb, all_item_embs) # [B,1,H]*[B,N,H]=[B,N,H]
        scores = torch.mean(scores, dim=-1) # [B,N] 

        scores, indices = torch.sort(scores, dim=-1, descending=True) # torch.sort https://hxhen.com/archives/226
        rank = torch.argwhere(indices==0)[:,1] # [B] rank最小为0 np.where https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
        isTop10 = (rank<10)
        isTop5 = (rank<5)
        isTop15 = (rank<15)
        logrank =  1/torch.log2(rank+2)
        
        HT5 = torch.sum(isTop5)
        HT10 = torch.sum(isTop10)
        HT15 = torch.sum(isTop15)
        NDCG5 = torch.sum(isTop5*logrank)
        NDCG10 = torch.sum(isTop10*logrank)
        NDCG15 = torch.sum(isTop15*logrank)
        return HT5, HT10, HT15, NDCG5, NDCG10, NDCG15

class Shadow_Model(nn.Module):
    
    def __init__(self, config, args, device):
        super(Shadow_Model, self).__init__()
        self.device = device
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.n_items = config['n_items']
        
        # Graph
        S = config['S']
        self.S = self.convert_numpy_to_tensor(S)
        A = config['A_tr']
        self.A = self.sparse_mx_to_torch_sparse_tensor(A)
        self.RRT = config['RRT_tr']
        self.RRTdrop = None
        
        # training hyper-parameter
        self.hidden = args.hidden
        self.neg = args.neg
        self.std = args.std 
        self.hop = args.hop
        self.drop = args.dropout
        self.decay = args.decay
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.recon_reg = args.recon_reg
        self.recon_drop = args.recon_drop
        self.kl_reg = args.kl_reg
        
        self.sigmoid = torch.nn.Sigmoid()
        
        # layer
        # self.user_embs = nn.Embedding(self.n_users, self.hidden) 
        # self.item_embs = nn.Embedding(self.n_items, self.hidden)
        # nn.init.xavier_uniform_(self.user_embs.weight)
        # nn.init.xavier_uniform_(self.item_embs.weight)
        
        self.user1_embs = nn.Embedding(self.n_users, self.hidden)
        self.item1_embs = nn.Embedding(self.n_items, self.hidden)
        self.user2_embs = nn.Embedding(self.n_users, self.hidden)
        self.item2_embs = nn.Embedding(self.n_items, self.hidden)
        nn.init.xavier_uniform_(self.user1_embs.weight)
        nn.init.xavier_uniform_(self.item1_embs.weight)
        nn.init.xavier_uniform_(self.user2_embs.weight)
        nn.init.xavier_uniform_(self.item2_embs.weight)
        
        # xavier_uniform nn.init.xavier_uniform_
        self.socialencoder = SocialGCN(hop=self.hop)
        self.uiencoder = InteractionGCN(hop=self.hop)
        
    
    def convert_numpy_to_tensor(self, adj):
        adj = torch.FloatTensor(adj).to(self.device)
        return adj

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        Graph = torch.sparse.FloatTensor(indices, values, shape)
        Graph = Graph.coalesce().to(self.device)
        return Graph
    
    def edge_dropout(self, sp_adj):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - self.recon_drop)))
        user1_np = np.array(row_idx)[keep_idx]
        user2_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user1_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user1_np, user2_np)), shape=(self.n_users, self.n_users))
        return dropped_adj


    def forward(self, users, pos, neg, epoch=0):
        """
        users:[B] pos:[B] neg:[B, neg] neg=1
        """
        all_user_embs_social = self.socialencoder(self.user1_embs.weight, self.S) # socialGCN
        all_item_embs_social = self.item1_embs.weight
        
        all_user_embs_rating, all_item_embs_rating = self.uiencoder(self.user2_embs.weight, self.item2_embs.weight, self.A) # ratingGCN
        
        users_emb_social = all_user_embs_social[users]
        pos_emb_social = all_item_embs_social[pos]
        neg_emb_social = all_item_embs_social[neg]

        users_emb_rating = all_user_embs_rating[users]
        pos_emb_rating = all_item_embs_rating[pos]
        neg_emb_rating = all_item_embs_rating[neg]
        
        return users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating
    
    def l2_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.mean(torch.pow(w, 2))

        return 0.5*loss

    def rec_loss(self, user_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(user_emb*pos_emb, dim=1)  # [B]
        neg_scores = torch.sum(user_emb*neg_emb, dim=1)  # [B]
        sup_logits = pos_scores-neg_scores
        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))
        return bpr_loss

    def calculate_loss(self, users, pos, neg, epoch):
        """
        Only this function appears in train()
        """
        users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating = self.forward(users.long(), pos.long(), neg.long(), epoch) # neg_emb:[B, d]
        # print(users_emb, pos_emb, neg_emb, all_user_embs_S, all_user_embs_A, \
        #     users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating)
        bpr_loss_social = self.rec_loss(users_emb_social, pos_emb_social, neg_emb_social)
        # print(bpr_loss_social)
        bpr_loss_rating = self.rec_loss(users_emb_rating, pos_emb_rating, neg_emb_rating)
        # print(bpr_loss_rating)
        
        reg_loss_social = self.l2_loss(
            self.user1_embs(users.long()),
            self.item1_embs(pos.long()),
            self.item1_embs(neg.long()),
        )
        reg_loss_rating = self.l2_loss(
            self.user2_embs(users.long()),
            self.item2_embs(pos.long()),
            self.item2_embs(neg.long()),
        )

        loss = bpr_loss_social+ bpr_loss_rating + self.decay*(reg_loss_social+ reg_loss_rating)
        # print('loss', loss)
        return loss

    def batch_full_sort_predict(self, users, pos, neg_items, epoch):
        users_emb_social, pos_emb_social, neg_emb_social, users_emb_rating, pos_emb_rating, neg_emb_rating = self.forward(users.long(), pos.long(), neg_items.long(), epoch)
        pos_emb_social = pos_emb_social.unsqueeze(1) # [B,1,H]
        all_item_embs_social = torch.cat([pos_emb_social, neg_emb_social], dim=1) # [B,N=1+999,H] cat方法按输入顺序拼接:先pos_emb, 再negs_emb
        users_emb_social = users_emb_social.unsqueeze(1) # [B,1,H]
        scores_social = torch.mul(users_emb_social, all_item_embs_social) # [B,1,H]*[B,N,H]=[B,N,H]
        scores_social = torch.mean(scores_social, dim=-1) # [B,N] 

        pos_emb_rating = pos_emb_rating.unsqueeze(1) # [B,1,H]
        all_item_embs_rating = torch.cat([pos_emb_rating, neg_emb_rating], dim=1) # [B,N=1+999,H] cat方法按输入顺序拼接:先pos_emb, 再negs_emb
        users_emb_rating = users_emb_rating.unsqueeze(1) # [B,1,H]
        scores_rating = torch.mul(users_emb_rating, all_item_embs_rating) # [B,1,H]*[B,N,H]=[B,N,H]
        scores_rating = torch.mean(scores_rating, dim=-1) # [B,N] 

        scores = (scores_rating + scores_social) / 2
        scores, indices = torch.sort(scores, dim=-1, descending=True) # torch.sort https://hxhen.com/archives/226
        rank = torch.argwhere(indices==0)[:,1] # [B] rank最小为0 np.where https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
        isTop10 = (rank<10)
        isTop5 = (rank<5)
        isTop15 = (rank<15)
        logrank =  1/torch.log2(rank+2)
        
        HT5 = torch.sum(isTop5)
        HT10 = torch.sum(isTop10)
        HT15 = torch.sum(isTop15)
        NDCG5 = torch.sum(isTop5*logrank)
        NDCG10 = torch.sum(isTop10*logrank)
        NDCG15 = torch.sum(isTop15*logrank)
        return HT5, HT10, HT15, NDCG5, NDCG10, NDCG15


class DiffNet(MyModel):
    
    def __init__(self, config, args, device):
        super(MyModel, self).__init__()
        self.device = device
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.n_items = config['n_items']
        self.RRT = config['RRT_tr']
        self.RRTdrop = None
        
        # Graph
        S = config['S']
        self.S = self.convert_numpy_to_tensor(S)
        A = config['A_tr']
        self.A = self.sparse_mx_to_torch_sparse_tensor(A)
        
        # training hyper-parameter
        self.hidden = args.hidden
        self.neg = args.neg
        self.std = args.std 
        self.hop = args.hop
        self.drop = args.dropout
        self.decay = args.decay

        # layer
        self.user_embs = nn.Embedding(self.n_users, self.hidden) 
        self.item_embs = nn.Embedding(self.n_items, self.hidden)
        nn.init.xavier_uniform_(self.user_embs.weight)
        nn.init.xavier_uniform_(self.item_embs.weight)
        # xavier_uniform nn.init.xavier_uniform_
        self.GCN_S = DiffNet_SocialGCN(hop=self.hop, drop=self.drop, hidden=self.hidden)
        self.GCN_A = DiffNet_InteractionGCN(hop=1)
    
    def convert_numpy_to_tensor(self, adj):
        adj = torch.FloatTensor(adj).to(self.device)
        return adj

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        Graph = torch.sparse.FloatTensor(indices, values, shape)
        Graph = Graph.coalesce().to(self.device)
        return Graph

    def edge_dropout(self, sp_adj):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - self.recon_drop)))
        user1_np = np.array(row_idx)[keep_idx]
        user2_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user1_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user1_np, user2_np)), shape=(self.n_users, self.n_users))
        return dropped_adj
    
    def forward(self, users, pos, neg, epoch):
        """
        users:[B] pos:[B] neg:[B, neg]
        """       
        all_user_embs_S = self.GCN_S(self.user_embs.weight, self.S) # tensor
        all_user_embs_A, all_item_embs_A = self.GCN_A(self.user_embs.weight, self.item_embs.weight, self.A) 
        
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A 
        users_emb = all_user_embs[users]
        pos_emb = all_item_embs_A[pos]
        neg_emb = all_item_embs_A[neg] # 经过print size 证实是[B,neg,H]
        # x[mask] 当mask为LongTensor https://blog.csdn.net/goodxin_ie/article/details/89672700 就跟nn.Embedding(input) 类似
        
        return users_emb, pos_emb, neg_emb
    
    def l2_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.mean(torch.pow(w, 2))

        return 0.5*loss

    def rec_loss(self, user_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(user_emb*pos_emb, dim=1)  # [B]
        neg_scores = torch.sum(user_emb*neg_emb, dim=1)  # [B]
        sup_logits = pos_scores-neg_scores
        bpr_loss = -torch.mean(F.logsigmoid(sup_logits))
        return bpr_loss    
    
    def calculate_loss(self, users, pos, neg, epoch):
        """
        Only this function appears in train()
        """
        (users_emb, pos_emb, neg_emb) = self.forward(users.long(), pos.long(), neg.long(), epoch)
        # users_emb:[B, d]  pos_emb:[B,d] neg_emb:[B, neg, d]
        bpr_loss = self.rec_loss(users_emb, pos_emb, neg_emb)
        
        reg_loss = self.l2_loss(
            self.user_embs(users.long()),
            self.item_embs(pos.long()),
            self.item_embs(neg.long()),
        )
        loss = bpr_loss + self.decay*reg_loss
        
        return loss
    
    def gen_uemb(self, users):
        all_user_embs_S = self.GCN_S(self.user_embs.weight, self.S) 
        all_user_embs_A, all_item_embs = self.GCN_A(self.user_embs.weight, self.item_embs.weight, self.A) 
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN

        return all_user_embs[users]
    
    def val_forward(self, users, G, mtd=0):
        """
        users:[B] pos:[B] neg:[B, neg] neg=1
        G 有可能修改过了
        """
        all_user_embs_S = self.GCN_S(self.user_embs.weight, self.S) 
        all_user_embs_A, all_item_embs = self.GCN_A(self.user_embs.weight, self.item_embs.weight, G) 
        all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN

        return all_user_embs[users], all_item_embs

    def gen_all_uemb(self):
        with torch.no_grad():
            all_user_embs_S = self.GCN_S(self.user_embs.weight, self.S) 
            all_user_embs_A, all_item_embs = self.GCN_A(self.user_embs.weight, self.item_embs.weight, self.A) 
            all_user_embs = 0.5*all_user_embs_S + 0.5*all_user_embs_A # twinGCN
            return {'user_embs.weight': all_user_embs}

    def batch_full_sort_predict(self, users, pos, neg_items, epoch):
        (user_emb, pos_emb, negs_emb) = self.forward(users.long(), pos.long(), neg_items.long(), epoch)
        pos_emb = pos_emb.unsqueeze(1) # [B,1,H]
        all_item_embs = torch.cat([pos_emb, negs_emb], dim=1) # [B,N=1+999,H] cat方法按输入顺序拼接:先pos_emb, 再negs_emb
        user_emb = user_emb.unsqueeze(1) # [B,1,H]
        scores = torch.mul(user_emb, all_item_embs) # [B,1,H]*[B,N,H]=[B,N,H]
        scores = torch.mean(scores, dim=-1) # [B,N] 

        scores, indices = torch.sort(scores, dim=-1, descending=True) # torch.sort https://hxhen.com/archives/226
        rank = torch.argwhere(indices==0)[:,1] # [B] rank最小为0 np.where https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
        isTop10 = (rank<10)
        isTop5 = (rank<5)
        isTop15 = (rank<15)
        logrank =  1/torch.log2(rank+2)
        
        HT5 = torch.sum(isTop5)
        HT10 = torch.sum(isTop10)
        HT15 = torch.sum(isTop15)
        NDCG5 = torch.sum(isTop5*logrank)
        NDCG10 = torch.sum(isTop10*logrank)
        NDCG15 = torch.sum(isTop15*logrank)
        return HT5, HT10, HT15, NDCG5, NDCG10, NDCG15

def create_model(config, args, device):
    if args.model_name == 'DESIGN':
        return DESIGN(config=config, args=args, device=device)
    elif args.model_name == 'DiffNet':
        return DiffNet(config=config, args=args, device=device)