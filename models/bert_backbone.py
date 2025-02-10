import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import XLNetModel, XLNetLMHeadModel, XLNetTokenizer
from transformers import XLNetForSequenceClassification
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from models.transformer.Layers import EncoderLayer


class Hi_XL_GCN(nn.Module):
    # args, s
    def __init__(self, args, negs, batch_size, margin, device, freeze_emb_layer=True):  # sen_embedding_matrix,
        super(Hi_XL_GCN, self).__init__()
        # sentence-level:
        self.alpha = args.alpha
        self.device = device
        # ('bert-{}-uncased'.format(model))
        self.sen_xl = BertModel.from_pretrained('bert-base-uncased')
        # lower transformer
        self.tf1 = BertModel.from_pretrained('bert-base-uncased')
        # higher transformer
        if args.model_size == 'base':
            hidden_size = 768
        elif args.model_size == 'large':
            hidden_size = 1024

        self.gc1 = GraphConvolution(args.hidden_dim, args.hidden_dim)
        self.gc2 = GraphConvolution(args.hidden_dim, args.hidden_dim)
        self.gc3 = GraphConvolution(args.hidden_dim, args.hidden_dim)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.layer_stack = nn.ModuleList([
            EncoderLayer(768, 512, 1, 512, 768, dropout=0.1)  # 512 1024 4 512 512 M
            for _ in range(1)])

        self.sentence_pooling = None

        self.end_token_id = tokenizer.sep_token_id

        if args.freeze_emb_layer and freeze_emb_layer:
            self.layer_freezing()

        self.margin = margin
        self.getTranspose = lambda x: torch.transpose(x, -2, -1)
        self.subMargin = lambda z: z - margin

        self.conlinear = nn.Linear(hidden_size, 1)

        self.mlp = nn.Linear(args.hidden_dim * 4, args.hidden_dim)

        self.sen_fc = nn.Linear(args.hidden_dim, args.polarities_dim)

    def getScore(self, doc, type):
        input_ids, attention_mask = doc
        batch_size = input_ids.shape[0]
        # how many sentences
        end_token_lookup = input_ids == self.end_token_id
        # get last layer sequence output
        outputs = self.tf1(input_ids=input_ids, output_hidden_states=True)
        output1 = outputs.last_hidden_state
        rep = outputs.pooler_output
        score = self.conlinear(rep).view(-1)

        prior_max_sent_count = end_token_lookup.sum(axis=-1).max()
        max_sent_count = max(prior_max_sent_count.item(), 1)
        # coherece sen_embeding
        sentence_embeddings = torch.zeros(batch_size, max_sent_count, self.tf1.config.hidden_size).to(self.device)
        sentence_attention_mask = torch.zeros(batch_size, max_sent_count, dtype=torch.long).to(self.device)
        for batch_idx in range(batch_size):
            local_sent_count = end_token_lookup[batch_idx].sum(axis=-1).item()
            if local_sent_count == 0:
                # if period is not present in the document use [CLS] token to mark whole document as one sentence
                local_sent_count = 1
                sentence_embeddings[batch_idx, :local_sent_count] = output1[batch_idx][0]
            else:
                # define different type of sub-word pooling for sentence embedding
                sentence_embeddings[batch_idx, :local_sent_count] = output1[batch_idx][end_token_lookup[batch_idx]]
            # sentence masking for next layer
            sentence_attention_mask[batch_idx, :local_sent_count] = 1

        if type == 'pos':
            return score, sentence_embeddings[0]
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

    def forward(self, inputs):  # label=None

        pos_input, neg_inputs, sen_input = inputs
        # sen_index, text_indices, sen_input_ids, sen_attention_mask, aspect_indices, adj = sen_input
        sen_index, cata_group, text_indices, sen_input_ids, sen_attention_mask, token_starts, token_starts_mask, left_indices, aspect_indices, catagory_indices, adj, aspect_in_text, aspect_in_text_mask = sen_input

        # keys = csd.values()

        pos_out, coherence_sen = self.getScore(pos_input, 'pos')

        neg_outs = []
        for inx in range(len(neg_inputs)):
            neg_outs.append(self.getScore(neg_inputs[inx], 'neg'))

        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        batch_size, max_len = text_indices.shape[0], text_indices.shape[1]
        outputs = self.sen_xl(input_ids=sen_input_ids, token_type_ids=None, attention_mask=sen_attention_mask,
                              output_hidden_states=True)
        stack_hidden_states = outputs.last_hidden_state
        sen_em = outputs.last_hidden_state[:, -1, :]
        hidden_states_list = []
        for i in range(batch_size):
            start_tokens_hidden_states = torch.index_select(stack_hidden_states[i, :], dim=0, index=token_starts[i])
            hidden_states_list.append(start_tokens_hidden_states)

        co_feature = torch.stack(hidden_states_list, dim=0)
        # feature = guidance_states
        x = F.relu(self.gc1(co_feature, adj))
        x = F.relu(self.gc2(x, adj))
        feature = F.relu(self.gc3(x, adj))
        # Aspect - aware mask
        x = self.mask(feature, aspect_double_idx)
        # Aspect - aware Attention

        alpha_mat = torch.matmul(x, co_feature.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, co_feature).squeeze(1)

        gcn_feature = self.alpha * x
        coherence_emb = torch.index_select(coherence_sen, 0, sen_index)

        group_len = torch.sum(cata_group != -1, dim=-1)
        group_emb = []

        for i in range(batch_size):
            groups = cata_group[i, 0:group_len[i]]
            group_emb_i = torch.index_select(coherence_emb, 0, groups)#.mean(dim=0)
            enc_output = group_emb_i.unsqueeze(0)
            for enc_layer in self.layer_stack:
                # residual = enc_output
                enc_output = enc_layer(
                    enc_output
                )
            enc_output = enc_output[0].squeeze(0).mean(dim=0)
            group_emb.append(enc_output)
            # group_emb.append(group_emb_i)

        coherece_group = torch.stack(group_emb, dim=0)
        # gcn_feature,
        sen_merged = torch.concat([sen_em,gcn_feature,coherece_group,coherence_emb], dim=-1)  # sen_em,gcn_feature,, coherece_group sen_em, ,
        sen_out = F.relu(self.mlp(sen_merged))
        #
        sen_fc = self.sen_fc(sen_out)

        return pos_out, neg_outs[0], sen_fc  # output2[:, 0, :]

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
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

