from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
import json
from torch.autograd import Variable
from torchvision import models, transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn import metrics
import os
import time
import apex
from apex import amp
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel
from apex.multi_tensor_apply import multi_tensor_applier
from torch.utils.data import Dataset, DataLoader as DL, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image 
import glob
import torch.distributed as dist
import torchvision
import random
import pandas as pd
import datetime
from thop import profile


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--path', default='/gputemp/FromCeph/Data/patches', type=str, help='path of patches')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--seed', default=99, type=int, help='seed')
    parser.add_argument('--data_set', default='HE', type=str)
    parser.add_argument('--mag', default='10', type=str)
    parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--momentum', default=0.90, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--lrdrop', default=150, type=int, help='multiply LR by 0.1 every (default: 150 epochs)')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--device', default='0,1,2,3', type=str)
    parser.add_argument('--comment', default='LNCR', type=str)
    parser.add_argument('--mod', default='inceptionV3', type=str)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()

args = get_parser()


class PedDataset(Dataset):
    def __init__(self, Data_path, ptids, Mag='10', transforms=None, limit=96):
        self.ptids = ptids
        self.slide = [
            (ptid, slide) 
            for ptid in ptids
            for slide in os.listdir(os.path.join(Data_path, ptid))
            if limit <= len(glob.glob(os.path.join(Data_path, ptid, slide, Mag, '*'))) and slide==data_set]
        
        index = 0
        self.patch = []
        self.label = []
        self.indices = {}
        for i, (ptid, slide) in enumerate(self.slide):
            patches = glob.glob(os.path.join(Data_path, ptid, slide, Mag, '*'))
            self.patch.extend(patches)
            label = data_map[ptid]['patient-label']
            self.label.extend([label] * len(patches))
            self.indices[(ptid, slide)] = np.arange(index, index+len(patches))
            index += len(patches)
        self.slide = np.array(self.slide)
        self.data_transforms = transforms
        
    def __len__(self):
        return len(self.patch)
    
    def __getitem__(self, index):
        img = Image.open(self.patch[index])
        label = self.label[index]
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, label

    
class DistSlideSampler(DistributedSampler):
    def __init__(self, dataset, batchsize, seed):
        super(DistSlideSampler, self).__init__(dataset)
        self.slide = dataset.slide
        self.indices = dataset.indices
        self.batchsize = batchsize
        self.seed = hash(seed)
        self.g = torch.Generator()
        
    def __iter__(self):
        self.g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            len(self.slide) - len(self.slide)%self.num_replicas, 
            generator=self.g
        ).tolist()
        for i in indices[self.rank::self.num_replicas]:
            ptid, slide = self.slide[i]
            yield self.get_slide(ptid, slide)
        
    def __len__(self):
        return len(self.slide) // self.num_replicas
    
    def get_slide(self, ptid, slide):
        indice = self.indices[(ptid, slide)]
        patch_num = len(indice)
        np.random.seed(self.seed % (2**32) + self.epoch)
        if patch_num <= self.batchsize:
            multip = self.batchsize // patch_num
            need = self.batchsize - multip*patch_num
            indice = np.concatenate(
                [indice]*multip +[np.random.choice(indice, need, replace=False)]
            ) 
        else:
            indice = np.random.choice(indice, self.batchsize, replace=False)
        shuffle = torch.randperm(self.batchsize, generator=self.g).tolist()
        return indice[shuffle]
        
    
class ValDistSlideSampler(DistributedSampler):
    def __init__(self, dataset, limit=512):
        super(ValDistSlideSampler, self).__init__(dataset)
        self.slide = dataset.slide
        self.indices = dataset.indices
        self.limit = limit
        
    def __len__(self):
        return len(self.slide) // self.num_replicas
    
    def __iter__(self):
        slide = self.slide[len(self.slide)%self.num_replicas:]
        for ptid, slide in slide[self.rank::self.num_replicas]:
            yield self.get_slide(ptid, slide)
            
    def get_slide(self, ptid, slide):
        indice = self.indices[(ptid, slide)]
        patch_num = len(indice)
        shuffle = torch.randperm(
            self.limit if patch_num > self.limit else patch_num
        ).tolist()
        return  indice[shuffle]
    

def fast_collate(batch, memory_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w, h = imgs[0].size[0], imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        numpy_array = np.asarray(img, dtype=np.uint8)
        numpy_array = np.rollaxis(numpy_array, 2)
        tensor[i] += torch.from_numpy(numpy_array.copy())
    return tensor, targets


def prepare_dataset(data_path, batchsize=64, mag='10', seed='None'):
    PATCHES_DATA_PATH = data_path
    limit = 2
    
    train_datasets = PedDataset(PATCHES_DATA_PATH, 
                                train_label, 
                                limit=limit, 
                                Mag=mag,  
                                transforms=train_transform)
    valid_datasets = PedDataset(PATCHES_DATA_PATH,  
                                val_label, 
                                limit=limit, 
                                Mag=mag, 
                                transforms=val_transform)
    
    if args.local_rank == 0:
        print('-'*30)
        print('Train slide number:', len(train_datasets.slide))
        print('Train pos ratio:', train_pos/len(train_datasets.slide))
        print('Train patches number:', len(train_datasets))
        print('Valid slide number:', len(valid_datasets.slide))
        print('Valid pos ratio:', valid_pos/len(valid_datasets.slide))
        print('Valid patches number:', len(valid_datasets))
        
    memory_format = torch.contiguous_format
    collate_fn = lambda b: fast_collate(b, memory_format)
    
    train_loader = DL(train_datasets, 
                      batch_sampler=DistSlideSampler(train_datasets, 
                                                     batchsize=batchsize, 
                                                     seed=seed),
                      num_workers=16,
                      pin_memory=True,
                      collate_fn=collate_fn
                     )
    valid_loader = DL(valid_datasets,
                      batch_sampler=ValDistSlideSampler(valid_datasets, 
                                                         limit=128),
                      num_workers=16,
                      pin_memory=True,
                      collate_fn=collate_fn
                      )
    return train_loader, valid_loader


class data_prefetcher():
    def __init__(self, loader, data_set='HE'):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        
        if data_set=='HE':
            self.mean = torch.tensor([0.7441 * 255, 0.5278 * 255, 0.7350 * 255]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.1951 * 255, 0.2996 * 255, 0.2122 * 255]).cuda().view(1,3,1,1)
        if data_set=='MASSON':
            self.mean = torch.tensor([0.6599 * 255, 0.5043 * 255, 0.6319 * 255]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.2691 * 255, 0.3343 * 255, 0.2675 * 255]).cuda().view(1,3,1,1)
        if data_set=='PAS':
            self.mean = torch.tensor([0.8474 * 255, 0.7170 * 255, 0.8241 * 255]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.1404 * 255, 0.2086 * 255, 0.1580 * 255]).cuda().view(1,3,1,1)
        if data_set=='PASM':
            self.mean = torch.tensor([0.4559 * 255, 0.3877 * 255, 0.4707 * 255]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.3390 * 255, 0.3588 * 255, 0.3665 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target
    

class Attention_Gated(nn.Module):
    def __init__(self, model, pretrain):
        super(Attention_Gated, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        
        if model == 'alexnet':
            self.feature_extractor = torchvision.models.alexnet(pretrained=True)
            self.feature_extractor.classifier = nn.Linear(9216, self.L)
            if args.local_rank == 0:
                input = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input, ))
                print('FLOPS:', flops)
                print('PARAMS:', params)
            nn.init.xavier_normal_(self.feature_extractor.classifier.weight)
        elif model == 'vgg11':
            self.feature_extractor = torchvision.models.vgg11(pretrained=True)
            self.feature_extractor.classifier = nn.Linear(25088, self.L)
            if args.local_rank == 0:
                input = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input, ))
                print('FLOPS:', flops)
                print('PARAMS:', params)
            nn.init.xavier_normal_(self.feature_extractor.classifier.weight)
        elif model == 'resnet50':
            self.feature_extractor = torchvision.models.resnet50(pretrained=True)
            self.feature_extractor.fc = nn.Linear(2048, self.L)
            if args.local_rank == 0:
                input = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input, ))
                print('FLOPS:', flops)
                print('PARAMS:', params)
            nn.init.xavier_normal_(self.feature_extractor.fc.weight)
        elif model == 'densenet121':
            self.feature_extractor = torchvision.models.densenet121(pretrained=True)
            self.feature_extractor.classifier = nn.Linear(1024, self.L)
            if args.local_rank == 0:
                input = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input, ))
                print('FLOPS:', flops)
                print('PARAMS:', params)
            nn.init.xavier_normal_(self.feature_extractor.classifier.weight)
        elif model == 'vit':
            import timm
            self.L = 128
            vit_model = timm.create_model('vit_base_patch16_384', pretrained=False)
            pretrained_model_path = '/group_homes/HCC_surv/home/share/vit_base_patch16_384_augreg_in21k_ft_in1k.bin'
            if pretrained_model_path is not None and os.path.isfile(pretrained_model_path):
                print("loading pretrain weights from :", pretrained_model_path)
                checkpoint = torch.load(pretrained_model_path)
                vit_model.load_state_dict(checkpoint, strict=False)  
            else:
                print("no such file, the model will train from scratch.")  
            self.feature_extractor = nn.Sequential(
                                        vit_model,
                                        nn.Linear(1000, 128)
                                    )
        else:
            self.feature_extractor = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
            self.feature_extractor.fc = nn.Linear(2048, self.L)
            if args.local_rank == 0:
                input = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input, ))
                print('FLOPS:', flops)
                print('PARAMS:', params)
            nn.init.xavier_normal_(self.feature_extractor.fc.weight)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.L, 
                                       nhead=8,
                                       activation='gelu'),
            num_layers=2,
            norm=nn.LayerNorm(normalized_shape=self.L, eps=1e-6)
        )
        
        self.attention = nn.Linear(self.L, self.K)
        nn.init.xavier_normal_(self.attention.weight)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        nn.init.xavier_normal_(self.classifier[0].weight)

    
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor(x)
        
        H = H.unsqueeze(0)
        H = self.encoder(H.transpose(0,1))
        H = H.transpose(0,1).squeeze()

        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        
        return Y_prob

def run(args, train_loader, valid_loader, model, epochs, schduler, optimizer, device, Writer):
    TrainAcc_list=[]
    TrainAUC_list=[]
    TrainLoss_list=[]
    EvalAcc_list=[]
    EvalAUC_list=[]
    EvalLoss_list=[]
           
    best_auc = .0
    for epoch in range(0, epochs):
        if args.local_rank == 0:
            print('Epoch [{}/{}]'.format(epoch, epochs))
            print('### Train ###')
        
        train_loader.batch_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            cut, TrainAcc, TrainAUC, TrainLoss = train_model(args, train_loader, model, device, optimizer, epoch, Writer)
        else:
            cut = train_model(args, train_loader, model, device, optimizer, epoch, Writer)
            
        schduler.step()
        current_lr = schduler.get_lr()[0]
        
        if args.local_rank == 0:
            print('### Valid ###')
        valid_loader.batch_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            EvalAcc, EvalAUC, EvalLoss = eval_model(args, valid_loader, model, device, optimizer, epoch, Writer, 'valid', cut)
        else:
            eval_model(args, valid_loader, model, device, optimizer, epoch, Writer, 'valid', cut)
        
        if args.local_rank == 0:
            TrainAcc_list.append(TrainAcc)
            TrainAUC_list.append(TrainAUC)
            TrainLoss_list.append(TrainLoss)
            EvalAcc_list.append(EvalAcc)
            EvalAUC_list.append(EvalAUC)
            EvalLoss_list.append(EvalLoss)

        ######################## Saving checkpoints and summary #########################
        if args.local_rank==0 and epoch >= 0:
            torch.save(model.state_dict(), 
                       os.path.join(
                           f'./Model_checkpoints/checkpoints_{mod}_{lr}_{batchsize}_{data_set}_{mag}_{Times}_{now}', 
                           args.comment,
                           '{}.pt'.format(epoch))
                       )
        ######################## Saving checkpoints and summary #########################
    
    if args.local_rank == 0:
        result = pd.DataFrame({
            'TrainAcc': TrainAcc_list,
            'TrainAUC': TrainAUC_list,
            'TrainLoss': TrainLoss_list,
            'EvalAcc': EvalAcc_list,
            'EvalAUC': EvalAUC_list,
            'EvalLoss': EvalLoss_list
            })
        result.to_csv(f'./Results/Result_{mod}_{lr}_{batchsize}_{data_set}_{mag}_{Times}_{now}.csv')

            
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt


def gather_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    var_list = [torch.zeros_like(rt) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(var_list, rt, async_op=False)
    return [i.item() for i in var_list]


def set_fn(v):
    def f(m):
        if isinstance(m, apex.parallel.SyncBatchNorm):
            m.momentum = v
    return f


def get_auc(ture, pred):
    fpr, tpr, thresholds = metrics.roc_curve(ture, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def train_model(args, dataloader, model, device, optimizer, epoch, Writer):
    phase = 'train'
    t=0.5
    model.train()
    
    all_labels = []
    all_values = []
    train_loss = 0
    
    prefetcher = data_prefetcher(dataloader, data_set)
    patches, label = prefetcher.next()
    index = 0
    while patches is not None:
        index += 1
        label = label[0]
        Y_prob= model.forward(patches)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.-1e-5)

        J = -1.*(
            label*torch.log(Y_prob)+
            (1.-label)*torch.log(1.-Y_prob)
        )
        
        optimizer.zero_grad()
        with amp.scale_loss(J, optimizer) as scale_loss:
            scale_loss.backward()
        optimizer.step()

        reduced_loss = reduce_tensor(J.data)
        train_loss += reduced_loss.item()
        
        all_labels.extend(gather_tensor(label))
        all_values.extend(gather_tensor(Y_prob[0][0]))
        
        patches, label = prefetcher.next()
        
    if args.local_rank == 0:
        print(len(all_labels))
        all_labels = np.array(all_labels)
        Loss = train_loss / len(all_labels)
        AUC, Acc, t = get_cm_train(all_labels, all_values)

        ######################## Saving checkpoints and summary #########################
        Writer.add_scalar('{}/Acc'.format(phase.capitalize()), Acc, epoch)
        Writer.add_scalar('{}/Loss'.format(phase.capitalize()), Loss, epoch)
        Writer.add_scalar('{}/Auc'.format(phase.capitalize()), AUC, epoch)
        ######################## Saving checkpoints and summary #########################
    
        return t, Acc, AUC, Loss
    
    else:
        return t


def eval_model(args, dataloader, model, device, optimizer, epoch, Writer, phase, t):
    model.eval()
    all_labels = []
    all_values = []
    eval_loss = 0
    
    prefetcher = data_prefetcher(dataloader, data_set)
    patches, label = prefetcher.next()
    index = 0
    while patches is not None:
        index += 1
        label = label[0].float()
        
        with torch.no_grad():
            Y_prob= model.forward(patches)
            Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

            J = -1.*(
                label*torch.log(Y_prob)+
                (1.-label)*torch.log(1.-Y_prob)
            )
        
        reduced_loss = reduce_tensor(J.data)
        
        eval_loss += reduced_loss.item()
        all_labels.extend(gather_tensor(label))
        all_values.extend(gather_tensor(Y_prob[0][0]))
        
        patches, label = prefetcher.next()
            
    if args.local_rank == 0:
        print(len(all_labels))
        all_labels = np.array(all_labels)
        Loss = eval_loss / len(all_labels)
        AUC, Acc = get_cm_val(all_labels, all_values, t)
        
        ######################## Saving checkpoints and summary #########################
        Writer.add_scalar('{}/Acc'.format(phase.capitalize()), Acc, epoch)
        Writer.add_scalar('{}/Auc'.format(phase.capitalize()), AUC, epoch)
        Writer.add_scalar('{}/Loss'.format(phase.capitalize()), Loss, epoch)
        ######################## Saving checkpoints and summary #########################

        return Acc, AUC, Loss
    
def get_cm_train(AllLabels, AllValues):
    fpr, tpr, threshold = roc_curve(AllLabels, AllValues, pos_label=1)
    Auc = auc(fpr, tpr)
    m = t = 0

    for i in range(len(threshold)):
        if tpr[i] - fpr[i] > m :
            m = abs(-fpr[i]+tpr[i])
            t = threshold[i]
    AllPred = [int(i>=t) for i in AllValues]
    Acc = sum([AllLabels[i] == AllPred[i] for i in range(len(AllPred))]) / len(AllPred)

    Pos_num = sum(AllLabels)
    Neg_num = len(AllLabels) - Pos_num
    cm = confusion_matrix(AllLabels, AllPred)
    print("[AUC/{:.4f}] [Threshold/{:.4f}] [Acc/{:.4f}]".format(Auc, t,  Acc))
    print("{:.2f}% {:.2f}%".format(cm[0][0]/ Neg_num * 100, cm[0][1]/Neg_num * 100))
    print("{:.2f}% {:.2f}%".format(cm[1][0]/ Pos_num * 100, cm[1][1]/Pos_num * 100))
    
    return Auc, Acc, t

def get_cm_val(AllLabels, AllValues, t):
    fpr, tpr, threshold = roc_curve(AllLabels, AllValues, pos_label=1)
    Auc = auc(fpr, tpr)
    
    AllPred = [int(i>=t) for i in AllValues]
    Acc = sum([AllLabels[i] == AllPred[i] for i in range(len(AllPred))]) / len(AllPred)

    Pos_num = sum(AllLabels)
    Neg_num = len(AllLabels) - Pos_num
    cm = confusion_matrix(AllLabels, AllPred)
    print("[AUC/{:.4f}] [Threshold/{:.4f}] [Acc/{:.4f}]".format(Auc, t,  Acc))
    print("{:.2f}% {:.2f}%".format(cm[0][0]/ Neg_num * 100, cm[0][1]/Neg_num * 100))
    print("{:.2f}% {:.2f}%".format(cm[1][0]/ Pos_num * 100, cm[1][1]/Pos_num * 100))
    
    return Auc, Acc


if __name__ == '__main__':
    
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method=args.init_method
    )
    name = args.comment
    now = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')

    train_transform = transforms.Compose([
            transforms.RandomCrop(384),
            transforms.Resize(299),
            ])
    val_transform = transforms.Compose([
            transforms.CenterCrop(384),
            transforms.Resize(299),
            ])
    if mod=='vit':
        train_transform = transforms.Compose([
            transforms.RandomCrop(384)
            ])
        test_transform = transforms.Compose([
            transforms.CenterCrop(384)
            ])
    
    with open(f'./Labels/Model.json3') as f:
    data_map = json.load(f)

    Model_slide_path = f'../pediatrics/Model_patches_normalized/'
    Model_Label_df = pd.read_excel('./Labels/Model_data.xlsx')
    train_label_ptid = list(map(str, Model_Label_df[Model_Label_df['group']==0]['ptid'].tolist()))
    val_label_ptid = list(map(str, Model_Label_df[Model_Label_df['group']==1]['ptid'].tolist())) 

    ###
    Model_data_avlb = []
    for sl in os.listdir(Model_slide_path):
        if os.path.exists(f'{Model_slide_path}/{sl}/{data_set}/{args.mag}'):
            if len(os.listdir(f'{Model_slide_path}/{sl}/{data_set}/{args.mag}')) > 1:
                Model_data_avlb.append(sl)

    train_label = list(set(train_label_ptid)&set(Model_data_avlb))
    val_label = list(set(val_label_ptid)&set(Model_data_avlb))
    train_label.sort()
    val_label.sort()
    
    ###
    writer = None
    if args.local_rank == 0:

        ######################## Saving checkpoints and summary #########################
        if os.path.exists(f'./Model_checkpoints/checkpoints_{mod}_{lr}_{batchsize}_{data_set}_{mag}_{Times}_{now}/comment/'):
            pass
        else:
            try:
                os.mkdir(f'./Model_checkpoints/checkpoints_{mod}_{lr}_{batchsize}_{data_set}_{mag}_{Times}_{now}/')
            except Exception:
                pass
            os.mkdir(f'./Model_checkpoints/checkpoints_{mod}_{lr}_{batchsize}_{data_set}_{mag}_{Times}_{now}/comment/')
        ######################## Saving checkpoints and summary #########################


        ######################## Saving checkpoints and summary #########################
        writer = SummaryWriter(f'./Model_runs/runs_{mod}_{lr}_{batchsize}_{data_set}_{mag}_{Times}_{now}/{name}')
        writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))
        ######################## Saving checkpoints and summary #########################

    train_loader, valid_loader = prepare_dataset(Model_slide_path, batchsize, mag, args.comment)
    device = torch.device(f"cuda:{args.local_rank}")

    model = apex.parallel.convert_syncbn_model(
        Attention_Gated(mod, args.pretrain)
    ).to(device)

    if mod=='vit': 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    model, optimizer = amp.initialize(model, optimizer, 
                                      opt_level="O0",
                                      keep_batchnorm_fp32=None)

    model = DistributedDataParallel(model, delay_allreduce=True)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs-5, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    run(args, train_loader, valid_loader, model, args.epochs, scheduler, optimizer, device, writer)





 
