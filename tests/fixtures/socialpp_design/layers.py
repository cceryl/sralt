import torch 
import torch.nn as nn
import torch.nn.functional as F

class SocialGCN(nn.Module):
    def __init__(self, hop):
        super(SocialGCN, self).__init__()
        self.hop = hop
    
    def forward(self, users_emb, adj):
        embs = [users_emb]
        for _ in range(self.hop):
            if adj.is_sparse:
                users_emb = torch.sparse.mm(adj, users_emb) # sparse x sparse -> sparse sparse x dense -> dense
            else:
                users_emb = torch.matmul(adj, users_emb)
            embs.append(users_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out

class InteractionGCN(nn.Module):
    def __init__(self, hop):
        super(InteractionGCN, self).__init__()
        self.hop = hop

    def forward(self, users_emb, items_emb, adj):
        num_users = users_emb.size()[0]
        num_items = items_emb.size()[0]
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(self.hop):
            # print(adj.shape, all_emb.shape)
            all_emb = torch.sparse.mm(adj, all_emb) # sparse x sparse -> sparse sparse x dense -> dense
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [num_users, num_items])
        return users, items
    

# 两个GraphLearner分别学两张图 forward:At=cosine similarity(Z) return At, f(At+A)
class GraphLearner(nn.Module):
    
    def __init__(self, input_size, topk, epsilon, num_pers, device, metric_type, graph_skip_conn):
        super().__init__()
        self.input_size = input_size # cosine similarity size
        self.topk = topk
        self.epsilon = epsilon
        self.num_pers = num_pers
        self.device = device
        self.metric_type = metric_type
        self.graph_skip_conn = graph_skip_conn
        if metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        
        
    def forward(self, feature, init_graph, epoch):
        if self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1) # [4, 1, feat_dim]
            # print("self.weight_tensor.size(): ", self.weight_tensor.size())
            context_fc = feature.unsqueeze(0) * expand_weight_tensor # [1, N, feat_dim]*[4, 1, feat_dim]=[4, N, feat_dim]
            # cosine_similarity(a,b)={a/|a|^2}*{b/|b|^2}
            context_norm = F.normalize(context_fc, p=2, dim=-1) # 指定维度上做均一化 v/|v|^2 
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0) # BB^T=[4, N, feat_dim][4, feat_dim, N]->[N, N]求similarity
            maskoff_value = 0   
        # graph structure learning 更新出的图本质上就是attention 每个位置上是对应权重
        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, maskoff_value)

        # if self.topk is not None:
        #     attention = self.build_knn_neighbourhood(attention, self.topk, maskoff_value)
        
        assert attention.min().item() >= 0
        
        # print(attention.size()) # [7317, 7317]
        
        # learned_graph = normalize_dense(attention)
        learned_graph = attention / torch.clamp(torch.sum(attention, dim=-1, keepdim=True), min=1e-12) # row-normalization 
        learned_graph = self.graph_skip_conn * init_graph + (1-self.graph_skip_conn) * learned_graph
        # graph_skip_conn = 1-min((epoch+1)/500, 1)
        # learned_graph = graph_skip_conn * init_graph + (1-graph_skip_conn) * learned_graph
        
        return attention, learned_graph

    def build_knn_neighbourhood(self, attention, topk, maskoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = (maskoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val).to(self.device)
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, maskoff_value):
        """
        Tensor.detach() is used to detach a tensor from the current computational graph. It returns a new tensor that doesn't require a gradient.

            When we don't need a tensor to be traced for the gradient computation, we detach the tensor from the current computational graph.

            We also need to detach a tensor when we need to move the tensor from GPU to CPU.
        """
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + maskoff_value * (1 - mask)
        return weighted_adjacency_matrix

import torch 
import torch.nn as nn
import torch.nn.functional as F

class DiffNet_InteractionGCN(nn.Module):
    def __init__(self, hop):
        super(DiffNet_InteractionGCN, self).__init__()
        self.hop = hop

    def forward(self, users_emb, items_emb, adj):
        num_users = users_emb.size()[0]
        num_items = items_emb.size()[0]
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(self.hop):
            all_emb = torch.sparse.mm(adj, all_emb) # sparse x sparse -> sparse sparse x dense -> dense
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [num_users, num_items])
        return users, items
    
class DiffNet_SocialGCN(nn.Module):
    def __init__(self, hop, drop, hidden) -> None:
        super().__init__()
        self.hop = hop
        self.hidden = hidden
        self.weight_dict = self.init_weight()
        self.dropout = drop
        self.tanh = nn.Tanh()
        
    def init_weight(self):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        for k in range(self.hop):
            weight_dict.update({'W_%d'%k: nn.Parameter(initializer(torch.empty(self.hidden*2,
                                                                      self.hidden)))})
        return weight_dict
    
    def forward(self, user_embs, adj):
        # adj = F.dropout(adj, p=self.dropout, training=self.training) # 必须training=self.training 只有nn.Dropout不需要这样显式声明
        for k in range(self.hop):
            new_user_embs = torch.matmul(adj, user_embs)
            user_embs = torch.matmul(torch.cat([new_user_embs, user_embs], dim=1), self.weight_dict['W_%d' %k])
            user_embs = self.tanh(user_embs)
            if k < self.hop-1:
                user_embs = F.dropout(user_embs, p=self.dropout, training=self.training)
        
        return user_embs
        