import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, num_features, num_hidden, num_class, dropout, alpha, num_heads):
        """
            Dense version of GAT.
            num_features: 特征维度
            num_hidden:
            num_class: 标签数量
            num_heads: 注意力头数量
            dropout: 置零比率
            alpha: 激活函数LeakyReLU的负轴斜率
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(num_features, num_hidden, dropout=dropout, alpha=alpha, concat=True) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(num_hidden * num_heads, num_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)