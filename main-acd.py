import torch
import load_acd
from models import hiacd
import time
import math
from torch.optim.swa_utils import SWALR
from args import parser
from tqdm import tqdm
import torch.nn as nn
import warnings
import optuna
from sklearn import metrics
import os
from math import isnan
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, hamming_loss, precision_score, recall_score
from torch.nn import CrossEntropyLoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

class TrainModel():

    def save_model(self, step, text):
        if not os.path.isdir('saved_models'):
            os.mkdir("saved_models")

        model_path = os.path.join("saved_models",
                                  "{}_seed-{}_lr-{}_{}_type-{}_p=5_all.pair".format(self.desc, self.seed,
                                                                                              self.learning_rate,
                                                                                              text,
                                                                                             self.model_size))
        print('model has saved')
        torch.save(self.xlnet_model.state_dict(), model_path)

    def account_implicite(self, t_outputs_all, t_targets_all, t_implicit):
        out_pre = torch.argmax(t_outputs_all, -1)
        im_num, ex_num = 0, 0
        im_cor, ex_cor = 0, 0
        for index in range(len(t_implicit)):
            all_num = len(t_implicit)
            if t_implicit[index] == 0:
                ex_num = ex_num + 1
                if t_targets_all[index] == out_pre[index]:
                    ex_cor = ex_cor + 1
            else:
                im_num = im_num + 1
                if t_targets_all[index] == out_pre[index]:
                    im_cor = im_cor + 1
        acc_im = im_cor / im_num
        acc_ex = ex_cor / ex_num
        print('account results: all : {}, im_num: {}, ex_num: {}, acc im : {}, acc ex {}'.format(all_num,im_num,ex_num,acc_im,acc_ex))
        return acc_im, acc_ex

    def account_aspect(self, t_outputs_all, t_targets_all, t_aspect):
        out_pre = torch.argmax(t_outputs_all, -1)
        has_num, no_num = 0, 0
        has_cor, no_cor = 0, 0
        all_num = len(t_aspect)
        for index in range(len(t_aspect)):
            # 'aspect==null': 0,
            # 'aspect is not null': 1,
            if t_aspect[index] == 0:
                no_num = no_num + 1
                if t_targets_all[index] == out_pre[index]:
                    no_cor = no_cor + 1
            else:
                has_num = has_num + 1
                if t_targets_all[index] == out_pre[index]:
                    has_cor = has_cor + 1

        acc_has = has_cor / has_num
        acc_no = no_cor / no_num
        print('account aspect results: all : {}, has_num: {}, no_num: {}, acc has : {}, acc no {}'.format(all_num, has_num, no_num, acc_has, acc_no))

    def _reset_params(self):
        for p in self.xlnet_model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def acc_and_f1(self, preds, labels):
        acc = self.simple_accuracy(preds, labels)
        precision = precision_score(labels, preds, average='micro')
        recall = recall_score(labels, preds, average='micro')
        f1 = f1_score(y_true=labels, y_pred=preds, average='micro')

        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "micro-f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def compute_metrics(self, preds, labels):
        assert len(preds) == len(labels)
        return self.acc_and_f1(preds, labels)

    def simple_accuracy(self, preds, labels):
        return (preds == labels).mean()

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.model_size = args.model_size
        self.learning_rate = args.lr_start
        self.a = args.a
        self.anneal_to = args.lr_end
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.negs = args.num_negs
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        if args.test_file:
            self.test_file = args.test_file
        else:
            self.test_file = args.dev_file

        self.margin = args.margin
        self.desc = args.model_description
        self.seed = args.seed
        self.datatype = args.data_type
        self.max_len = args.max_len
        self.bestacc = 0
        self.bestacc_sen = 0

        self.train_data = load_acd.LoadConnData(self.train_file, self.batch_size, self.model_size, self.device,
                                                 self.datatype, self.max_len)
        self.dev_data = load_acd.LoadConnData(self.dev_file,self.batch_size, self.model_size, self.device,
                                               self.datatype, self.max_len) #
        self.test_data = load_acd.LoadConnData(self.test_file,self.batch_size, self.model_size, self.device,
                                               self.datatype, self.max_len) #
        self.xlnet_model = hiacd.Hi_XL_ACD(args, self.negs, self.batch_size, self.margin, self.device)

        # self.criterion = nn.CrossEntropyLoss()
        self.cate_loss_fct = BCEWithLogitsLoss()
        # self.cate_loss_fct = BCEWithLogitsLoss(reduction="sum")

        self.xlnet_model = self.xlnet_model.to(self.device)
        self.initializer = args.initializer

        optimizer_grouped_parameters = [
            {'params': self.xlnet_model.sen_xl.parameters(), 'weight_decay': 0.001, 'lr': 2e-5},
            {'params': self.xlnet_model.tf1.parameters(), 'weight_decay':  0.00001, 'lr': args.lr_start},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)  # , lr=self.learning_rate
        # self.scheduler = SWALR(self.optimizer, anneal_strategy="linear", anneal_epochs=args.lr_anneal_epochs,
        #                        swa_lr=args.lr_end)
        self.total_loss = 0.0
        self.total_sen_loss = 0.0
        self.total_doc_loss = 0.0
        self.total_cat_loss = 0.0
        self.eval_interval = args.eval_interval

    def train_xlnet_model(self):
        train_loader = self.train_data.data_loader()
        self.epochs = 20
        start = time.time()
        self.xlnet_model.train()
        ddd = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]

        for epoch in range(self.epochs):
            n_correct, n_total = 0, 0
            print("Epoch: {}/{}".format(epoch + 1, self.epochs))
            for step, data in enumerate(tqdm(train_loader)):
                try:
                    pos_input, neg_input, sentence = data
                except Error as e:
                    print(e)
                    continue

                sentence_inputs = [sentence['sen_index'][0, :].to(self.device),
                                   sentence['catagory'][0, :].to(self.device),
                                   sentence['is_masked'][0, :].to(self.device),
                                   sentence['text_indices'][0, :].to(self.device),
                                   sentence['input_ids'][0, :].to(self.device),
                                   sentence['attention_mask'][0, :].to(self.device),
                                   sentence['token_starts'][0, :].to(self.device),
                                   sentence['token_start_mask'][0, :].to(self.device),
                                   sentence['dependency_graph'][0, :].to(self.device),]

                category_label = sentence['category_label_id'][0, :].to(self.device)
                sentiment_label = sentence['sentiment_label_ids'][0, :].to(self.device),
                pos_input = [pos_input[0]['pos_input_ids'].to(self.device),
                             pos_input[0]['pos_attention_mask'].to(self.device)]

                neg_inputs = []
                for i in range(len(neg_input)):
                    neg_inputs.append([neg_input[i]['neg_input_ids'].to(self.device),neg_input[i]['neg_attention_mask'].to(self.device)])

                pos_score, neg_scores, category_logits, sentiment_logits = self.xlnet_model([pos_input, neg_inputs, sentence_inputs])
                # cate_loss_fct = BCEWithLogitsLoss() masked_logits,
                # mask_loss = 0.01*self.cate_loss_fct(masked_logits, category_label.view(-1, category_label.shape[-1]).float())
                cat_loss = 0.1*self.cate_loss_fct(category_logits, category_label.view(-1, category_label.shape[-1]).float())

                sm = torch.nn.Softmax(dim=-1)
                final_sentiment_logits = - torch.log(sm(sentiment_logits))
                final_sentiment_logits = final_sentiment_logits * sentiment_label[0].float()

                sen_loss = torch.mean(torch.sum(final_sentiment_logits, dim=-1))

                docu_loss =self.xlnet_model.contrastiveLoss(pos_score, neg_scores)

                loss = cat_loss + sen_loss + docu_loss #+ cl_loss  # sa_loss +
                # loss = sen_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.xlnet_model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.total_sen_loss += sen_loss.item()
                self.total_doc_loss += docu_loss.item()
                self.total_cat_loss += cat_loss.item()
                self.total_loss += loss.item()

                category_logits = category_logits.detach().cpu().numpy()
                sentiment_logits = sentiment_logits.detach().cpu().numpy()
                logits = []
                logits_c = []

                for i in range(len(category_logits)):
                    tmp_c = []
                    tmp = []
                    for j in range(len(category_logits[i])):
                        if category_logits[i][j] < 2:
                            tmp = np.append(tmp, ddd[-1])
                            tmp_c = np.append(tmp_c, 0)
                        else:
                            tmp_c = np.append(tmp_c, 1)
                            tmp = np.append(tmp, ddd[np.argmax(sentiment_logits[i][j])])

                    logits.append(tmp)
                    logits_c.append(tmp_c)

                logits_c = torch.tensor(logits_c).int().cuda()
                n_correct += (logits_c == category_label).sum().item()

                sentiment_labels = sentiment_label[0].view(-1, sentiment_label[0].shape[0])
                sentiment_labels = sentiment_labels.transpose(1,0)

                n_total += len(category_logits)
                train_acc = n_correct / n_total

                if isnan(self.total_loss)==True:
                    # print(self.total_loss)
                    break

                if step % self.eval_interval == 0 and step > 0:
                    bestacc,_ =self.eval_model(step, start)
                    # self.scheduler.step()
                    print("Steps: {} total Loss: {} sen loss: {} cl loss: {}, doc loss: {},train sen Acc: {}".format(step, self.total_loss,
                                                                                                             self.total_sen_loss,
                                                                                                             self.total_cat_loss,
                                                                                                             self.total_doc_loss,train_acc))
                    self.total_loss = 0.0
                    self.total_sen_loss = 0.0
                    self.total_doc_loss = 0.0
                    self.total_cl_loss = 0.0

        return self.bestacc_sen

    def eval_model(self, step, start):
        dev_loader = self.test_data.data_loader()
        self.xlnet_model.eval()
        correct = 0.0
        total = 0.0

        ddd = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]

        t_targets_all, t_outputs_all, t_sentiment_outs, t_sentiment_labels = None, None, None, None
        self.y_pres = []
        self.y_tures = []

        index = 0
        selected =[16, 24]

        with torch.no_grad():
            for data in dev_loader:
                # index = index+1

                try:
                    pos_input, neg_input, sentence = data
                except Error as e:
                    print(e)
                    continue
                # sentence -level training
                sentence_inputs = [sentence['sen_index'][0, :].to(self.device),
                                   sentence['catagory'][0, :].to(self.device),
                                   sentence['is_masked'][0, :].to(self.device),
                                   sentence['text_indices'][0, :].to(self.device),
                                   sentence['input_ids'][0, :].to(self.device),
                                   sentence['attention_mask'][0, :].to(self.device),
                                   sentence['token_starts'][0, :].to(self.device),
                                   sentence['token_start_mask'][0, :].to(self.device),
                                   sentence['dependency_graph'][0, :].to(self.device),]

                category_label_ids = sentence['category_label_id'][0, :].to(self.device)
                sentiment_label_ids = sentence['sentiment_label_ids'][0, :].to(self.device),
                sentiment_label_ids = sentiment_label_ids[0]

                pos_input = [pos_input[0]['pos_input_ids'].to(self.device), pos_input[0]['pos_attention_mask'].to(self.device)]
                neg_inputs = []
                for i in range(len(neg_input)):
                    neg_inputs.append([neg_input[i]['neg_input_ids'].to(self.device),
                                       neg_input[i]['neg_attention_mask'].to(self.device)])

                if index in selected:
                    print(selected)

                pos_score, neg_scores, category_logits, sentiment_logits = self.xlnet_model([pos_input, neg_inputs, sentence_inputs])

                category_logits = category_logits.detach().cpu().numpy()
                sentiment_logits = sentiment_logits.detach().cpu().numpy()

                logits = []
                logits_c = []

                for i in range(len(category_logits)):
                    tmp = []
                    tmp_c = []
                    for j in range(len(category_logits[i])):
                        if category_logits[i][j] < 2:
                            tmp = np.append(tmp, ddd[-1])
                            tmp_c = np.append(tmp_c, 0)
                        else:
                            tmp = np.append(tmp, ddd[np.argmax(sentiment_logits[i][j])])
                            tmp_c = np.append(tmp_c, 1)

                    logits.append(tmp)
                    logits_c.append(tmp_c)

                logits_c = torch.tensor(logits_c).int().cuda()
                logits = torch.tensor(logits).int().cuda()

                sentiment_label_ids = torch.argmax(sentiment_label_ids, dim=-1)
                c_label_ids = []
                s_label_ids = []

                for i in range(len(category_label_ids)):
                    tmp = []
                    tmp_c = []
                    for j in range(len(category_label_ids[i])):
                        if category_label_ids[i][j] == 0:
                            tmp = np.append(tmp, ddd[-1])
                            tmp_c = np.append(tmp_c, 0)
                        else:
                            tmp = np.append(tmp, ddd[sentiment_label_ids[i][j]])
                            tmp_c = np.append(tmp_c, 1)
                    c_label_ids.append(tmp_c)
                    s_label_ids.append(tmp)

                c_label_ids = torch.tensor(c_label_ids).int().cuda()
                s_label_ids = torch.tensor(s_label_ids).int().cuda()

                max_neg_score = torch.max(neg_scores, -1).values
                if pos_score > max_neg_score:
                    correct += 1.0
                total += 1.0

                # t_sentiment_outs, t_sentiment_labels
                if t_targets_all is None:
                    t_targets_all = c_label_ids
                    t_outputs_all = logits_c
                    t_sentiment_outs = logits
                    t_sentiment_labels = s_label_ids
                else:
                    t_targets_all = torch.cat((t_targets_all, c_label_ids), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, logits_c), dim=0)
                    t_sentiment_outs = torch.cat((t_sentiment_outs, logits), dim=0)
                    t_sentiment_labels = torch.cat((t_sentiment_labels, s_label_ids), dim=0)

        self.xlnet_model.train()
        acc = correct / total

        result = self.compute_metrics(t_outputs_all.detach().cpu().numpy(), t_targets_all.detach().cpu().numpy())
        print("category", result)

        result1 = self.compute_metrics(t_sentiment_outs.detach().cpu().numpy(), t_sentiment_labels.detach().cpu().numpy())
        print("sentiment", result1)
        f1 = result['micro-f1']

        if step > 0:
            if f1 > self.bestacc_sen and f1 > 0.777:
                self.bestacc_sen = f1
                self.desc = 'lap16_acsa'
                print("saved")
                text ="ACC: "+ '%.4f'%self.bestacc_sen+ "F1: "+ '%.4f' % f1
                print(self.desc)
                print(text)
                self.save_model(step,text)
        return self.bestacc_sen, acc

def main(trial):
    opt = parser.parse_args()
    # # optuna setting for tuning hyperparameters
    # opt.alpha = trial.suggest_uniform('alpha', 0.6, 0.9)
    # opt.a = trial.suggest_int('a', 0.1, 0.9)
    # opt.lr_start = trial.suggest_uniform('lr_start', 0.000005, 0.00001)
    # opt.alpha = trial.suggest_uniform('alpha', 0.6, 0.9)
    # opt.alpha = trial.suggest_int('d_model', 0.1, 0.9, 0.05)
    # opt.lr_start = trial.suggest_uniform('lr_start', 0.000005, 0.00001)
    # opt.lr_end = trial.suggest_uniform('lr_end', 0.00001, 0.00005)
    # opt.dropout = trial.suggest_uniform('dropout_rate', 0.1, 0.9)
    initializers = {
        # xavier_uniform_
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    opt.initializer = initializers[opt.initializer]
    print('[Info] parameters: {}'.format(opt))

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    start = time.time()
    Trainer = TrainModel(opt)
    bestacc_sen = Trainer.train_xlnet_model()
    with open('result.txt', 'a') as f:
        f.write('[Info] parameters: {} \n'.format(opt))
        f.write(str(bestacc_sen)+"\n")
        f.close()
    return bestacc_sen

if __name__ == '__main__':

    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=5)
    df = study.trials_dataframe()

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))