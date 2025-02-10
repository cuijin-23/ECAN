import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetModel, XLNetLMHeadModel, XLNetTokenizer
from transformers import XLNetForSequenceClassification
from transformers import AutoModel, AutoTokenizer, AutoConfig
from models.transformer.Layers import EncoderLayer
import math
import numpy as np

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class Hi_XL_ACD(nn.Module):

    def __init__(self, args, negs, batch_size, margin, device, freeze_emb_layer=True):  # sen_embedding_matrix,
        super(Hi_XL_ACD, self).__init__()
        # sentence-level:
        self.alpha = args.alpha
        self.device = device
        self.sen_xl = XLNetModel.from_pretrained('xlnet-base-cased')
        # lower transformer
        self.tf1 = XLNetModel.from_pretrained('xlnet-base-cased')
        # higher transformer
        if args.model_size == 'base':
            hidden_size = 768
        elif args.model_size == 'large':
            hidden_size = 1024

        num_labels = 22
        self.num_labels = [num_labels, 3]
        self.attention_c = nn.ModuleList([self_attention_layer(hidden_size) for _ in range(num_labels)])
        # self.attention_intra_c = nn.ModuleList([self_attention_layer(hidden_size) for _ in range(num_labels)])
        # self.attention_s = nn.ModuleList([self_attention_layer(hidden_size) for _ in range(self.num_labels[1])])

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

        self.dropout = nn.Dropout(args.dropout_rate)
        self.dropout_1 = nn.ModuleList([nn.Dropout(args.dropout_rate) for _ in range(num_labels)])
        # self.SentiNorm = nn.ModuleList([LayerNorm(hidden_size) for _ in range(3)])
        # self.CateNorm = nn.ModuleList([LayerNorm(hidden_size) for _ in range(3)])
        self.layernorm = LayerNorm(hidden_size)

        self.layer_stack_c = nn.ModuleList([
            EncoderLayer(768, 750, 7, 750, 768, dropout=0.1)  # 512 1024 4 512 512 M
            for _ in range(num_labels)])

        self.layer_stack_s = nn.ModuleList([
            EncoderLayer(768, 750, 4, 750, 768, dropout=0.1)  # 512 1024 4 512 512 M
            for _ in range(self.num_labels[1])])

        self.dense_output = nn.ModuleList([nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh(),
        ) for _ in range(2)])

        self.dense_outputs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
        ) for _ in range(2)])

        self.linear_weight = nn.Linear(self.tf1.config.hidden_size, self.tf1.config.hidden_size)
        self.linear_value = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(self.tf1.config.hidden_size, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                                         requires_grad=True)
        self.sentence_pooling = None
        self.end_token_id = tokenizer.sep_token_id
        if args.freeze_emb_layer and freeze_emb_layer:
            self.layer_freezing()
        self.margin = margin
        self.getTranspose = lambda x: torch.transpose(x, -2, -1)
        self.subMargin = lambda z: z - margin
        self.conlinear = nn.Linear(hidden_size, 1)
        self.syn_map = nn.Sequential(
            nn.Flatten(),  # 将输入展平成n*115200维度
            nn.Linear(args.hidden_dim * 150, args.hidden_dim), # 映射层将输入特征映射到输出特征
            nn.ReLU(),
        )
        self.mlp = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.classifier_cate = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(num_labels)])
        # self.classifier_pot_cate = nn.Linear(hidden_size, args.polarities_dim)
        self.classifier_senti = nn.ModuleList([nn.Linear(hidden_size, self.num_labels[1]) for i in range(num_labels)])
        self.before = nn.Linear(args.hidden_dim, num_labels)
        # self.final = nn.Linear(args.polarities_dim * 2, args.polarities_dim)
        self.mlp_cate = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

        self.sen_mlp = nn.Sequential(
            nn.Linear((self.num_labels[1]+1)*args.hidden_dim, num_labels*args.hidden_dim),  # 输入维度为32*25*300，输出维度为32*22*300
            nn.ReLU(),
        )
        # self.sen_mlp = nn.Sequential(
        #     nn.Linear((self.num_labels[1])*args.hidden_dim, num_labels*args.hidden_dim),  # 输入维度为32*25*300，输出维度为32*22*300
        #     nn.ReLU(),
        # )
        self.multi_map = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        # +self.num_labels[0]
        self.gc1 = GraphConvolution(args.hidden_dim, args.hidden_dim)
        self.gc2 = GraphConvolution(args.hidden_dim, args.hidden_dim)
        self.gc3 = GraphConvolution(args.hidden_dim, args.hidden_dim)


        self.attention_pooling = nn.Linear(args.hidden_dim, 1)
        nn.init.xavier_uniform_(self.attention_pooling.weight)

    #
    # def _get_sentences_offsets(self, sentence_sep_tokens_pos):
    #     sentence_offsets = []
    #     start_index = 1
    #     sorted_indexes, _ = torch.nonzero(sentence_sep_tokens_pos).squeeze(1).sort()
    #     for index in sorted_indexes:
    #         end_index = index.item() - 1
    #         sentence_offsets.append([start_index, end_index])
    #         start_index = index.item() + 1
    #     return sentence_offsets


    def _get_sentences_offsets(self, sentence_sep_tokens_pos):
        sentence_offsets = []
        start_index = 0
        sorted_indexes, _ = torch.nonzero(sentence_sep_tokens_pos).squeeze(1).sort()
        for index in sorted_indexes:
            # if  index.item() == index.item()
            end_index = index.item()
            if start_index < end_index:
                sentence_offsets.append([start_index, end_index])
            else:
                sentence_offsets
            start_index = index.item()+1
        return sentence_offsets

    def getScore(self, doc, type):
        input_ids, attention_mask = doc
        batch_size = input_ids.shape[0]
        # how many sentences
        end_token_lookup = input_ids == self.end_token_id
        # get last layer sequence output
        output1 = self.tf1(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # 整体的语义 DS [cls]
        rep = output1[:, -1, :]
        score = self.conlinear(rep).view(-1)
        prior_max_sent_count = end_token_lookup.sum(axis=-1).max()
        max_sent_count = max(prior_max_sent_count.item(), 1)
        # coherece sen_embeding
        sentence_embeddings = torch.zeros(batch_size, max_sent_count, self.tf1.config.hidden_size).to(self.device)
        sentence_c = torch.zeros(max_sent_count,self.num_labels[0], self.tf1.config.hidden_size).to(self.device)
        sentence_s = torch.zeros(max_sent_count, self.num_labels[1], self.tf1.config.hidden_size).to(self.device)

        for batch_idx in range(batch_size):
            local_sent_count = end_token_lookup[batch_idx].sum(axis=-1).item()
            if local_sent_count == 0:
                # if period is not present in the document use [CLS] token to mark whole document as one sentence
                local_sent_count = 1
                sentence_embeddings[batch_idx, :local_sent_count] = output1[batch_idx][0]
            else:
                with torch.no_grad():
                    sentences_offsets = self._get_sentences_offsets(end_token_lookup[batch_idx])
                for sent_indx in range(local_sent_count):
                    start_index, end_index = sentences_offsets[sent_indx]
                    # print(sentences_offsets[sent_indx])
                    enc_output = output1[batch_idx][start_index:end_index]
                    temp, _ = torch.max(output1[batch_idx][start_index:end_index],axis=0)
                    # temp = torch.mean(output1[batch_idx][start_index:end_index],axis=0)
                    enc_output = enc_output.unsqueeze(0)
                    # before = self.before(enc_output)
                    multi_head_c = [F.relu(self.layer_stack_c[i](enc_output)[0].squeeze(0)) for i in
                                    range(self.num_labels[0])]

                    # multi_head_s = [multi_head_c[i] for i in range(self.num_labels[0])]
                    cc = [torch.mean(self.attention_pooling(multi_head_c[i])*multi_head_c[i], dim=0) for i in range(self.num_labels[0])]
                    ss = [self.layer_stack_s[i](enc_output)[0].squeeze(0).max(dim=0)[0] for i in range(self.num_labels[1])]
                    sentence_c[sent_indx] = torch.cat([torch.unsqueeze(cc[i], 0) for i in range(self.num_labels[0])], dim=0)
                    sentence_s[sent_indx] = torch.cat([torch.unsqueeze(ss[i], 0) for i in range(self.num_labels[1])], dim=0)
                    sentence_embeddings[batch_idx, sent_indx] = temp

        if type == 'pos':
            return score, sentence_embeddings[0], sentence_c, sentence_s
        else:
            return score

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.device)
        return mask * x

    def forward(self, inputs, category_map):  # label=None
        pos_input, neg_inputs, sen_input = inputs
        sen_index, cata_group, is_masked, text_indices, sen_input_ids, sen_attention_mask, token_starts, token_starts_mask, adj = sen_input
        pos_out, coherence_sen, co_c_feature, co_s_feature = self.getScore(pos_input, 'pos')

        batch_size = text_indices.shape[0]
        neg_outs = []
        for inx in range(len(neg_inputs)):
            neg_outs.append(self.getScore(neg_inputs[inx], 'neg'))

        outputs = self.sen_xl(input_ids=sen_input_ids, token_type_ids=None, attention_mask=sen_attention_mask,
                              output_hidden_states=True)

        coherence_emb = torch.index_select(coherence_sen, 0, sen_index)
        coherence_c_feature = torch.index_select(co_c_feature, 0, sen_index)
        coherence_s_feature = torch.index_select(co_s_feature, 0, sen_index)
        # coherence_c_feature = self.layernorm(coherence_c_feature)
        # coherence_s_feature = self.layernorm(coherence_s_feature)

        category_pooled_output = [self.attention_c[i](outputs.last_hidden_state) for i in range(self.num_labels[0])]
        # cc = [self.layer_stack_c[i](outputs.last_hidden_state)[0].squeeze(0).mean(dim=0) for i in range(self.num_labels[0])]
        # category_pooled_output = [self.dropout_1[0](category_pooled_output[i]) for i in range(self.num_labels[0])]
        # sentiment_emb = self.dropout_1[1](outputs.last_hidden_state[:, -1, :])

        stack_hidden_states = self.dropout(outputs.last_hidden_state)
        hidden_states_list = []
        for i in range(batch_size):
            start_tokens_hidden_states = torch.index_select(stack_hidden_states[i, :], dim=0, index=token_starts[i])
            hidden_states_list.append(start_tokens_hidden_states)

        co_feature = torch.stack(hidden_states_list, dim=0)
        # feature = guidance_states
        x = F.relu(self.gc1(co_feature, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        # sen_feature = self.syn_map(x)

        alpha_mat = torch.matmul(x, stack_hidden_states.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        sen_feature = torch.matmul(alpha, stack_hidden_states).squeeze(1)

        sen_merged = torch.concat([sen_feature, coherence_emb], dim=-1)  # masked_outputs, coherece_group, coherence_emb coherence_emb
        sen_merged = F.relu(self.mlp(sen_merged))
        # ++++++++++++++++++++
        # sen_merged = coherence_emb
        # ++++++++++++++++++++
        c_feature = torch.cat([torch.unsqueeze(category_pooled_output[i], 1) for i in range(self.num_labels[0])], dim=1)
        # c_feature = self.layernorm(c_feature)

        final_cate_feature = torch.zeros(len(c_feature),self.num_labels[0], self.tf1.config.hidden_size).to(c_feature.device)

        # attention pooliing
        for sent_indx in range(len(c_feature)):
            temp = coherence_c_feature[sent_indx]
            subword_rep = torch.tanh(self.linear_weight(temp))
            attention_weights = torch.softmax(subword_rep.mm(self.linear_value), dim=0)
            final_cate_feature[sent_indx] = torch.sum((attention_weights * c_feature[sent_indx]), dim=0)
        # ++++++++++++++++++++
        s_feature = torch.unsqueeze(sen_merged, 1)
        # ++++++++++++++++++++
        # s_feature = s_feature.repeat(1, self.num_labels[1], 1)
        # final_s_feature = torch.zeros(len(s_feature),self.num_labels[1], self.tf1.config.hidden_size).to(s_feature.device)
        # for sent_indx in range(len(s_feature)):
        #     subword_rep = torch.tanh(self.linear_weight(co_s_feature[0]))
        #     attention_weights = torch.softmax(subword_rep.mm(self.linear_value), dim=0)
        #     final_s_feature[sent_indx] = torch.sum((attention_weights * s_feature[sent_indx]), dim=0)

        final_feature = torch.cat([coherence_s_feature,s_feature], dim=1) #coherence_feature s_feature,
        # final_feature = self.SentiNorm[0](final_feature)
        # final_feature = final_feature.repeat(1, self.num_labels[0], 1)
        x = final_feature.view(final_feature.size(0), -1)
        final_features = self.sen_mlp(x)
        final_features = final_features.view(final_features.size(0), self.num_labels[0], 768)

        category_logits = torch.cat([self.classifier_cate[i]((final_cate_feature[:, i, :])) for i in range(self.num_labels[0])], dim=-1)
        sentiment_logits = torch.cat([self.classifier_senti[i](torch.unsqueeze(final_features[:, i, :], 1)) for i in range(self.num_labels[0])], dim=1)

        return pos_out, neg_outs[0], category_logits, sentiment_logits  # output2[:, 0, :] masked_logit

    def pairwiseLoss(self, pos_score, neg_score):
        zero_tensor = torch.zeros_like(pos_score)
        margin_tensor = torch.tensor(self.margin).cuda()
        loss_tensor = margin_tensor + neg_score - pos_score
        loss = torch.max(zero_tensor, loss_tensor)
        loss = torch.mean(loss)
        return loss

    def contrastiveLoss(self, pos_score, neg_scores):
        neg_scores_sub = torch.stack(list(map(self.subMargin, neg_scores)))
        all_scores = torch.cat((neg_scores_sub, pos_score), dim=-1)
        lsmax = -1 * F.log_softmax(all_scores, dim=-1)
        pos_loss = lsmax[-1]
        return pos_loss

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bigas
        else:
            return output

class self_attention_layer(nn.Module):
    def __init__(self, n_hidden):
        """
        Self-attention layer
        * n_hidden [int]: hidden layer number (equal to 2*n_hidden if bi-direction)
        """
        super(self_attention_layer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_hidden, 1, bias=False)
        )

    def init_weights(self):
        """
        Initialize all the weights and biases for this layer
        """
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.02)
                nn.init.uniform_(m.bias, -0.02, 0.02)

    def forward(self, inputs, mask=None):
        """
        Forward calculation of the layer
        * inputs [tensor]: input tensor (batch_size * max_seq_len * n_hidden)
        * seq_len [tensor]: sequence length (batch_size,)
        - outputs [tensor]: attention output (batch_size * n_hidden)
        """
        if inputs.dim() != 3 :
            raise ValueError("! Wrong dimemsion of the inputs parameters.")

        now_batch_size, max_seq_len, _ = inputs.size()
        alpha = self.attention(inputs).contiguous().view(now_batch_size, 1, max_seq_len)
        exp = torch.exp(alpha)

        if mask is not None:
            # mask = get_mask(inputs, seq_len)
            # mask = mask.contiguous().view(now_batch_size, 1, max_seq_len)
            mask = mask.unsqueeze(1)
            exp = exp * mask.float()

        sum_exp = exp.sum(-1, True) + 1e-9
        softmax_exp = exp / sum_exp.expand_as(exp).contiguous().view(now_batch_size, 1, max_seq_len)
        outputs = torch.bmm(softmax_exp, inputs).squeeze(-2)
        return outputs

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias