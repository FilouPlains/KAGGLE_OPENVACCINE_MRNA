import random
import numpy as np
import torch 
import torch.optim as optim
import argparse
from utils.bert import get_config, BertModel, BertForMaskedLM
from dataload import DATA 
import datetime
from collections import OrderedDict


random.seed(10)
torch.manual_seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser(description='RNABERT')
parser.add_argument('--mag',  type=int, default=1,
                    help='enumerate')
parser.add_argument('--epoch', '-e', type=int, default=200,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--batch', '-b', type=int, default=20,
                    help='Number of batch size')
parser.add_argument('--maskrate', '-m', type=float, default=0.0,
                    help='mask rate')
parser.add_argument('--pretraining', '-pre', type=str, help='use pretrained weight')
parser.add_argument('--algorithm', type=str, default="global", help='algorithm method')
parser.add_argument('--data_embedding', type=str, nargs='*', help='data for base embedding')
parser.add_argument('--embedding_output', type=str, nargs='*', help='output file for base embedding')
parser.add_argument('--show_aln', action='store_true')

args = parser.parse_args()
batch_size = args.batch
current_time = datetime.datetime.now()

print("start...")
class TRAIN:
    """The class for controlling the training process of SFP"""
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def model_device(self, model):
        print("device: ", self.device)
        print('-----start-------')
        model.to(self.device)
        if self.device == 'cuda':
            model = torch.nn.DataParallel(model) # make parallel
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return model

    # make feature vector 
    def make_feature(self, model, dataloader, seqs):
        model.eval()
        torch.backends.cudnn.benchmark = True
        batch_size = dataloader.batch_size
        encoding = []
        for batch in dataloader:
            data, label, seq_len= batch
            inputs = data.to(self.device)
            prediction_scores, prediction_scores_ss, encoded_layers =  model(inputs)
            encoding.append(encoded_layers.cpu().detach().numpy())
        encoding = np.concatenate(encoding,axis=0)
        print(np.shape(encoding))
        embedding = []
        for e, seq in zip(encoding, seqs):
            embedding.append(e[:len(seq)].tolist())

        return embedding 

# Here the code was modified to be able to read the pre train model
def objective():
    config.hidden_size = config.num_attention_heads * config.multiple    
    train = TRAIN(config)
    model = BertModel(config)
    model = BertForMaskedLM(config, model)
    model = train.model_device(model)
    # create new OrderedDict that does not contain `module.`
    if args.pretraining:
        save = torch.load(args.pretraining, map_location=train.device)
        new_state_dict = OrderedDict()
        for k, v in save.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Pretrain success")
    optimizer = optim.AdamW([{'params': model.parameters(), 'lr': config.adam_lr}])
    return model , optimizer, train, config

config = get_config(file_path = "./RNA_bert_config.json")
data = DATA(args, config)
model, optimizer, train, config = objective()



if args.data_embedding:
    seqs, label, test_dl  = data.load_data_EMB(args.data_embedding) 
    features = train.make_feature(model, test_dl, seqs)
    for i, data_set in enumerate(args.embedding_output):
        with open(data_set, 'w') as f:
            for d in features:
                f.write(str(d) + '\n')