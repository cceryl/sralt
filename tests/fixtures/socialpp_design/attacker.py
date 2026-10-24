import torch
import torch.nn as nn
import torch.nn.functional as F

# attacker
class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super(Discriminator, self).__init__()
        self.embed_dim = int(embed_dim)

        self.out_dim = 1   
        self.activation = torch.sigmoid
        self.drop = 0.5
        self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )
        
    def forward(self, ents_emb):
        scores = self.net(ents_emb)
        output = self.activation(scores)
        return output

    def predict(self, ents_emb):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = self.activation(scores)
            preds = (scores > torch.Tensor([0.5]).to(ents_emb.device)).float() * 1
            return output.squeeze(), preds

class EnDiscriminator(nn.Module):
    def __init__(self, feat_size, wide_size):
        super(EnDiscriminator, self).__init__()
        self.wide_size = int(wide_size)
        self.feat_size = int(feat_size)

        self.out_dim = 1   
        self.activation = torch.sigmoid
        self.drop = 0.5
        self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(int(self.feat_size), int(self.feat_size), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.feat_size), int(self.feat_size / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.feat_size / 2), self.out_dim, bias=True)
        )
        
        # raw user emb --> de ui info user emb
        self.wide_layer = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(int(self.wide_size), int(self.wide_size), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.wide_size), int(self.wide_size), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.wide_size), int(self.out_dim), bias=True)
        )

    def forward(self, ents_emb):
        wide_emb, deep_emb = ents_emb
        deep_s = self.net(deep_emb)
        wide_s = self.wide_layer(wide_emb)
        scores = wide_s + deep_s
        output = self.activation(scores)
        return output

    def predict(self, ents_emb):
        wide_emb, deep_emb = ents_emb
        with torch.no_grad():
            wide_emb, deep_emb = ents_emb
            deep_s = self.net(deep_emb)
            wide_s = self.wide_layer(wide_emb)
            scores = wide_s + deep_s
            output = self.activation(scores)
            preds = (scores > torch.Tensor([0.5]).to(wide_emb.device)).float() * 1
            return output.squeeze(), preds

# 跟attacker结构稍微不一致
class Defender(nn.Module):
    def __init__(self, embed_dim):
        super(Defender, self).__init__()
        self.embed_dim = int(embed_dim)

        self.out_dim = 1   
        self.activation = torch.sigmoid
        self.drop = 0.5
        self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

    def forward(self, ents_emb):
        scores = self.net(ents_emb)
        output = torch.sigmoid(scores)
        return output

    def predict(self, ents_emb):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = self.activation(scores)
            preds = (scores > torch.Tensor([0.5]).to(ents_emb.device)).float() * 1
            return output.squeeze(), preds
