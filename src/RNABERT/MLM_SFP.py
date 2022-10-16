import random
import time
import numpy as np
import torch 
import torch.optim as optim
from torchvision import transforms, datasets
import argparse
from utils.bert import get_config, BertModel, set_learned_params, BertForMaskedLM, visualize_attention, show_base_PCA, fix_params
from module import Train_Module
from dataload import DATA, MyDataset 
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering, SpectralClustering 
import itertools  
from collections import OrderedDict
import alignment_C as Aln_C

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
parser.add_argument('--outputweight', type=str, help='output path for weights')
parser.add_argument('--algorithm', type=str, default="global", help='algorithm method')
parser.add_argument('--data_mlm', '-d', type=str, nargs='*', help='data for mlm training')
parser.add_argument('--data_mul', type=str, nargs='*', help='data for mul training')
parser.add_argument('--data_alignment', type=str, nargs='*', help='data for alignment test')
parser.add_argument('--data_clustering', type=str, nargs='*', help='data for clustering test')
parser.add_argument('--data_showbase', type=str, nargs='*', help='data for base embedding')
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
        self.module = Train_Module(config)
    
    def model_device(self, model):
        print("device: ", self.device)
        print('-----start-------')
        model.to(self.device)
        if self.device == 'cuda':
            model = torch.nn.DataParallel(model) # make parallel
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return model

    def train_MLM_SFP(self, model, optimizer, dl_MLM_SFP, num_epochs, task_type):
        for epoch in range(num_epochs):
            model.train()
            epoch_mlm_loss = 0.0
            epoch_ssl_loss = 0.0
            epoch_mlm_correct = 0.0
            epoch_ssl_correct = 0.0
            epoch_sfp_loss=0.0
            epoch_sfp_correct = 0.0
            epoch_mul_loss = 0.0

            iteration = 1
            t_epoch_start = time.time()
            t_iter_start = time.time()
            data_num = 0
            for batch in dl_MLM_SFP:
                optimizer.zero_grad()
                if task_type == "MLM" or task_type == "SFP":
                    low_seq_0, masked_seq_0, family_0, seq_len_0, low_seq_1, masked_seq_1, family_1, seq_len_1 = batch
                elif task_type == "MUL":
                    low_seq_0, masked_seq_0, family_0, seq_len_0, low_seq_1, masked_seq_1, family_1, seq_len_1, common_index_0, common_index_1 = batch

                masked_seq_0 = masked_seq_0.to(self.device)
                low_seq_0 = low_seq_0.to(self.device)
                masked_seq_1 = masked_seq_1.to(self.device)
                low_seq_1 = low_seq_1.to(self.device)

                masked_seq = torch.cat((masked_seq_0, masked_seq_1), axis=0) 
                prediction_scores, prediction_scores_ss, encoded_layers =  model(masked_seq)
                prediction_scores0, prediction_scores1 = torch.split(prediction_scores, int(prediction_scores.shape[0]/2))
                prediction_scores_ss0, prediction_scores_ss1 = torch.split(prediction_scores_ss, int(prediction_scores_ss.shape[0]/2))
                encoded_layers0, encoded_layers1 = torch.split(encoded_layers, int(encoded_layers.shape[0]/2))

                loss = 0
                # MLM LOSS
                mlm_loss_0, mlm_correct_0 = self.module.train_MLM(low_seq_0, masked_seq_0, prediction_scores0)
                mlm_loss_1, mlm_correct_1 = self.module.train_MLM(low_seq_1, masked_seq_1, prediction_scores1)
                mlm_loss = (mlm_loss_0 + mlm_loss_1)/2
                mlm_loss = torch.tensor(0.0) if  torch.isnan(mlm_loss) else mlm_loss 
                mlm_correct = (mlm_correct_0 + mlm_correct_1)/2
                epoch_mlm_loss += mlm_loss.item() * batch_size
                epoch_mlm_correct += mlm_correct
                if task_type == "MLM":    
                    loss += mlm_loss

                # SFP LOSS
                if task_type == "SFP":    
                    z0_list, z1_list =  self.module.em(encoded_layers0, seq_len_0), self.module.em(encoded_layers1, seq_len_1)
                    sfp_loss, sfp_correct = self.module.train_SFP(low_seq_0, seq_len_0, low_seq_1, seq_len_1, family_0, family_1, z0_list, z1_list)
                    sfp_loss = torch.tensor(0.0) if  torch.isnan(sfp_loss) else sfp_loss 
                    epoch_sfp_loss += sfp_loss.item()* batch_size
                    epoch_sfp_correct += sfp_correct
                    loss += sfp_loss

                # MULTIPLE LOSS
                if task_type == "MUL":
                    common_index_0 = common_index_0.to(self.device)
                    common_index_1 = common_index_1.to(self.device)
                    z0_list, z1_list =  self.module.em(encoded_layers0, seq_len_0), self.module.em(encoded_layers1, seq_len_1)
                    mul_loss = self.module.train_MUL(z0_list, z1_list, common_index_0, common_index_1, seq_len_0, seq_len_1)
                    mul_loss = torch.tensor(0.0) if  torch.isnan(mul_loss) else mul_loss 
                    epoch_mul_loss += mul_loss.item()
                    loss +=  mul_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            t_epoch_finish = time.time()
            epoch_mlm_loss = epoch_mlm_loss / len(dl_MLM_SFP.dataset)
            epoch_mlm_correct = epoch_mlm_correct / len(dl_MLM_SFP)
            epoch_sfp_loss = epoch_sfp_loss  / len(dl_MLM_SFP.dataset)
            epoch_sfp_correct = epoch_sfp_correct / len(dl_MLM_SFP.dataset)
            epoch_mul_loss = epoch_mul_loss
            print('Epoch {}/{} | MLM Loss: {:.4f} MLM Acc: {:.4f}| SFP Loss: {:.4f} SFP Acc: {:.4f}| MUL Loss: {:.4f}| time: {:.4f} sec.'.format(epoch+1, num_epochs,
                                                                        epoch_mlm_loss, epoch_mlm_correct, epoch_sfp_loss, epoch_sfp_correct, epoch_mul_loss, time.time() - t_epoch_start))
            t_epoch_start = time.time()
        if args.outputweight:
            torch.save(model.state_dict(), args.outputweight + '{0:%m_%d_%H_%M}'.format(current_time))
            torch.save(model.state_dict(), args.outputweight)
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


def objective():
    config.hidden_size = config.num_attention_heads * config.multiple    
    train = TRAIN(config)
    model = BertModel(config)
    model = BertForMaskedLM(config, model)
    if args.data_mlm:
        config.adam_lr = 2e-4
    # if args.data_sfp:
    #     model = fix_params(model)
    #     config.adam_lr = config.adam_lr * 0.5
    if args.data_mul:
        # model = fix_params(model)
        config.adam_lr = 1e-4
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