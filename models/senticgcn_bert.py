# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dynamic_rnn import DynamicLSTM
from transformers import AutoModel

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SenticGCN_BERT(nn.Module):
    def __init__(self, opt, device):
        super(SenticGCN_BERT, self).__init__()
        self.opt = opt
        self.device = device
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        #self.dropout = nn.Dropout(opt.dropout)
        #self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        #self.text_lstm = DynamicLSTM(768, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gc2 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gc3 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        #self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc5 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc6 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc7 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc8 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        #batch_size = len(x)
        #seq_len = len(x[1])
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1,self.opt.max_len)):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, self.opt.max_len)):
                mask[i].append(1)
            for j in range(min(aspect_double_idx[i,1]+1, self.opt.max_len), seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.device)
        return mask*x

    def forward(self, inputs):
        text_indices, text_bert_indices, aspect_indices, bert_segments_ids, left_indices, adj = inputs

        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        #text = self.embed(text_indices)
        #text = self.text_embed_dropout(text)
        #text_out, (_, _) = self.text_lstm(text, text_len)

        encoder_layer, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_hidden_states=False, return_dict=False)

        text_out = encoder_layer

        # x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        # x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        # x = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc4(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc5(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc6(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc7(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc8(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))

        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output

class ASGCN(nn.Module):
    def __init__(self, opt, embedding_matrix, device):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.hidden_dim = 300
        self.device = device
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        # self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.text_lstm = DynamicLSTM(opt.embed_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        self.gc1 = GraphConvolution(self.hidden_dim,  self.hidden_dim)
        self.gc2 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)

            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
            # if aspect == null
            mask[i] = mask[i][0:seq_len]
        # mask =
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.device)

        return mask*x

    def cut_a(self, adj,text_len):
        batchsize = len(adj)
        max_len = max([ t for t in text_len])
        res = torch.zeros(batchsize,max_len,max_len)
        for i in range(batchsize):
            res[i] = adj[i][0:max_len,0:max_len]
        res = res.to(self.device)
        return res

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1).cpu()

        adj = self.cut_a(adj,text_len)

        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)

        # x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        # x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, adj))
        # x = F.relu(self.gc3(x, adj))
        x = self.mask(x, aspect_double_idx)

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output
