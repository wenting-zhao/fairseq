import sys
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
import numpy as np
from sklearn.metrics import accuracy_score

class Probe(nn.Module):
    # Cumulative scoring probe for CLS token
    def __init__(self, L, D_in = 1024, H = 1024):
        # L = probe level
        super().__init__()
        assert L <= 24
        self.L = L
        self.H = H
        self.probe_layers = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, 1)
            )
        self.mixing = nn.Parameter(torch.rand(L+1))
        self.scalar = nn.Parameter(torch.rand(1))

    def forward(self, cls_embs):
        # compute weighted sum of embeddings
        cls_embs = cls_embs.permute(0,2,1)
        weighted_cls = self.scalar * (F.log_softmax(self.mixing, dim=-1) * cls_embs).permute(0,2,1)
        weighted_cls = torch.sum(weighted_cls, dim=1)
        return self.probe_layers(weighted_cls)

def get_examples(infile, labels):
    f = open(infile, 'r')
    xs = [line.strip() for line in f]
    #for i in range(len(xs)):
    #    xs[i] = [int(j) for j in xs[i].split()]
    f.close()
    f = open(labels, 'r')
    ys = [line.strip() for line in f]
    assert len(xs) == len(ys)
    return xs, ys

def train_probe(dataset, m, d):
    model = Probe(24)
    model.cuda()
    optim = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-6)
    crit = nn.BCEWithLogitsLoss()
    bs = 128
    train_xs = dataset['train']['src']
    train_ys = torch.Tensor(dataset['train']['tgt'])
    valid_xs = dataset['dev']['src']
    valid_ys = torch.Tensor(dataset['dev']['tgt'])
    test_xs = dataset['test']['src']
    test_ys = torch.Tensor(dataset['test']['tgt'])
    best_valid_acc = 0
    batches = list(range(0, len(train_ys), bs))
    valid_batches = range(0, len(valid_ys), bs)
    test_batches = range(0, len(test_ys), bs)
    for i in range(50):
        print("epoch", i)
        print('learning rate:', optim.param_groups[0]['lr'])
        out_len = 1
        all_predictions = np.zeros(len(train_ys))
        all_targets = np.zeros(len(train_ys))
        tot_loss = 0.
        j = 0
        random.shuffle(batches)
        model.train()
        for i in tqdm(batches, mininterval=0.5,desc='(Training)', leave=False):
            start_idx, end_idx = i,min(i+bs, len(train_ys))
            xs = train_xs[start_idx:end_idx].cuda()
            ys = train_ys[start_idx:end_idx].cuda()
            optim.zero_grad()
            outputs = model(xs)
            outputs = outputs.squeeze()
            loss = crit(outputs, ys)
            tot_loss += loss.item()
            loss.backward()
            optim.step()
            all_predictions[start_idx:end_idx] = outputs.cpu().data.numpy()
            all_targets[start_idx:end_idx] = ys.cpu().data.numpy()
        all_predictions[all_predictions<0.5] = 0
        all_predictions[all_predictions>=0.5] = 1
        train_acc = accuracy_score(all_targets, all_predictions)
        print("train loss:", tot_loss/len(batches))
        print("train acc:", train_acc) 
        valid_loss = 0.
        model.eval()
        with torch.no_grad():
            all_predictions = np.zeros(len(train_ys))
            all_targets = np.zeros(len(train_ys))
            for i in tqdm(valid_batches, mininterval=0.5,desc='(Validating)', leave=False):
                start_idx, end_idx = i,min(i+bs, len(valid_ys))
                xs = valid_xs[start_idx:end_idx].cuda()
                ys = valid_ys[start_idx:end_idx].cuda()
                outputs = model(xs)
                outputs = outputs.squeeze()
                loss = crit(outputs, ys)
                valid_loss += loss.item()
                all_predictions[start_idx:end_idx] = outputs.cpu().data.numpy()
                all_predictions[all_predictions<0.5] = 0
                all_predictions[all_predictions>=0.5] = 1
                all_targets[start_idx:end_idx] = ys.cpu().data.numpy()
            valid_acc = accuracy_score(all_targets, all_predictions)
            print("valid loss:", valid_loss/len(valid_batches))
            print("valid acc:", valid_acc) 
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                print("found better model!")
                torch.save(model.state_dict(), "out/{}_{}_best_checkpoint.pt".format(m, d))
                test_loss = 0.
                all_predictions = np.zeros(len(train_ys))
                all_targets = np.zeros(len(train_ys))
                for i in tqdm(test_batches, mininterval=0.5,desc='(Testing)', leave=False):
                    start_idx, end_idx = i,min(i+bs, len(valid_ys))
                    xs = test_xs[start_idx:end_idx].cuda()
                    ys = test_ys[start_idx:end_idx].cuda()
                    outputs = model(xs)
                    outputs = outputs.squeeze()
                    loss = crit(outputs, ys)
                    test_loss += loss.item()
                    all_predictions[start_idx:end_idx] = outputs.cpu().data.numpy()
                    all_targets[start_idx:end_idx] = ys.cpu().data.numpy()
                all_predictions[all_predictions<0.5] = 0
                all_predictions[all_predictions>=0.5] = 1
                test_acc = accuracy_score(all_targets, all_predictions)
                print("test loss:", test_loss/len(test_batches))
                print("test acc:", test_acc) 




def get_clsemb(model, dataset):
    ckp = sys.argv[3]
    datapath = sys.argv[4]
    p = sys.argv[5]
    head_name = sys.argv[6]
    roberta = RobertaModel.from_pretrained(ckp, checkpoint_file='checkpoint_best.pt', data_name_or_path=datapath)
    roberta.cuda()
    roberta.eval()
    label_fn = lambda label: roberta.task.label_dictionary.string([label + roberta.task.label_dictionary.nspecial])
    #test_examples, test_ys = get_test_examples('ruletaker/test.input0.bpe', 'rawrule/d0/test.label')

    #for i in tqdm(batches):
    #    xs = test_examples[i:i+bs]
    #    ys = test_ys[i:i+bs]
    #    xbatch = collate_tokens([roberta.encode(test_examples[j]) for j in range(i, min(len(test_ys), i+bs))], pad_idx=1)
    #    pred = label_fn(roberta.predict('ruletaker_head', xbatch)).argmax(dim=1).cpu().data.numpy()
    #    correct_cnt += np.sum(np.array(ys) == np.array(pred))
    for split in ['train', 'dev', 'test']:
        examples, ys = get_examples(os.path.join(p, '%s.input0'%split), os.path.join(p, '%s.label'%split))
        print("loaded data from", os.path.join(p, '%s.input0'%split), "and", os.path.join(p, '%s.label'%split))
        correct_cnt = 0
        #bs = 8
        #batches = range(0, len(test_examples), bs)
        out = torch.zeros((len(ys), 25, 1024))
        preds = []
        with torch.no_grad():
            for i in tqdm(range(len(examples)), desc=split):
                x = examples[i]
                y = ys[i]
                tokens = roberta.encode(x)
                pred = label_fn(roberta.predict(head_name, tokens).argmax().item())
                preds.append(pred)
                if pred == y: correct_cnt += 1
                features = roberta.extract_features(tokens, return_all_hiddens=True)
                cls_embs = torch.zeros((len(features), 1024))
                for k in range(len(features)):
                    cls_embs[k] = features[k][0,0,:]
                out[i] = cls_embs.cpu()
        print("acc is", correct_cnt/len(ys))
        torch.save(out, 'out/{0}_{1}_{2}_embs.pt'.format(model, dataset, split)) 
        torch.save(preds, 'out/{0}_{1}_{2}_preds.pt'.format(model, dataset, split))
        torch.save([int(xx) for xx in ys], 'out/{0}_{1}_{2}_labels.pt'.format(model, dataset, split))

def main():
    model = sys.argv[1]
    dataset = sys.argv[2]
    if not os.path.isfile('out/{0}_{1}_test_embs.pt'.format(model, dataset)):
        get_clsemb(model, dataset)
    data = dict()
    data['train'], data['dev'], data['test'] = dict(), dict(), dict()
    for split in ['train', 'dev', 'test']:
        data[split]['src'] = torch.load('out/{0}_{1}_{2}_embs.pt'.format(model, dataset, split))
        data[split]['tgt'] = torch.load('out/{0}_{1}_{2}_labels.pt'.format(model, dataset, split))
    train_probe(data, model, dataset)

if __name__ == '__main__':
    main()
