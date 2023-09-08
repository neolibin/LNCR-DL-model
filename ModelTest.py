from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
from torch.autograd import Variable
from torchvision import models, transforms
import torch.nn.functional as F
import pandas as pd
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import random
from PIL import Image 
from sklearn import metrics
import glob
import os
import time
import argparse
import json
import apex
from apex import amp
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel
from apex.multi_tensor_apply import multi_tensor_applier
from torch.utils.data import Dataset, DataLoader as DL, Sampler
from torch.utils.data.distributed import DistributedSampler
import torchvision
import pandas as pd
from thop import profile


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--testpath', default='None', type=str)
    parser.add_argument('--modelpath', default='None', type=str)
    parser.add_argument('--data_set', default='HE', type=str)
    parser.add_argument('--mag', default='10', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', default=99, type=int, help='seed')
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()

args = get_parser()

# # ###
test_transform = transforms.Compose([
            transforms.CenterCrop(384),
            transforms.Resize(299),
        ])


class PedDataset(Dataset):
    def __init__(self, Data_path, ptids, Mag='5', transforms=None, limit=96):
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
            if Data_path==Model_slide_path:
                label = Model_data_map[ptid]['patient-label']
            if Data_path==test_slide_path:
                label = test_data_map[ptid]['patient-label']
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
        
    
class TestDistSlideSampler(DistributedSampler):
    def __init__(self, dataset, limit=512):
        super(TestDistSlideSampler, self).__init__(dataset)
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

def prepare_dataset(data_path, mag='10', seed='None'):
    Test_DATA_PATH = data_path
    PatchLimit = 128
    
    test_datasets = PedDataset(Test_DATA_PATH,  
                                test_label, 
                                limit=limit, 
                                Mag=mag, 
                                transforms=test_transform)
    
        
    memory_format = torch.contiguous_format
    collate_fn = lambda b: fast_collate(b, memory_format)
    
    test_loader = DL(test_datasets,
                     batch_sampler=TestDistSlideSampler(test_datasets, 
                                                     limit=PatchLimit),
                     num_workers=16, 
                     pin_memory=True, 
                     collate_fn=collate_fn
                    )
    
    return test_datasets, test_loader

class data_prefetcher():
    def __init__(self, loader, dataset='train'):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        
        self.mean = torch.tensor([0.6522 * 255, 0.3254 * 255, 0.6157 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.2044 * 255, 0.2466 * 255, 0.1815 * 255]).cuda().view(1,3,1,1)
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
            self.feature_extractor = torchvision.models.alexnet(pretrained=False)
            self.feature_extractor.classifier = nn.Linear(9216, self.L)
            if args.local_rank == 0:
                input_test = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input_test, ))
                print('FLOPS:', flops)
                print('PARAMS:', params)
            nn.init.xavier_normal_(self.feature_extractor.classifier.weight)
        elif model == 'vgg11':
            self.feature_extractor = torchvision.models.vgg11(pretrained=False)
            self.feature_extractor.classifier = nn.Linear(25088, self.L)
            if args.local_rank == 0:
                input_test = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input_test, ))
                print('FLOPS:', flops)
                print('PARAMS:', params)
            nn.init.xavier_normal_(self.feature_extractor.classifier.weight)
        elif model == 'resnet50':
            self.feature_extractor = torchvision.models.resnet50(pretrained=False)
            self.feature_extractor.fc = nn.Linear(2048, self.L)
            if args.local_rank == 0:
                input_test = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input_test, ))
                print('FLOPS:', flops)
                print('PARAMS:', params)
            nn.init.xavier_normal_(self.feature_extractor.fc.weight)
        elif model == 'densenet121':
            self.feature_extractor = torchvision.models.densenet121(pretrained=True)
            self.feature_extractor.classifier = nn.Linear(1024, self.L)
            if args.local_rank == 0:
                input_test = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input_test, ))
                print('FLOPS:', flops)
                print('PARAMS:', params)
            nn.init.xavier_normal_(self.feature_extractor.classifier.weight)
        else:
            self.feature_extractor = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
            self.feature_extractor.fc = nn.Linear(2048, self.L)
            if args.local_rank == 0:
                input_test = torch.randn(1, 3, 224, 224)
                flops, params = profile(self.feature_extractor, inputs=(input_test, ))
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
        
        
class Attention_Gated_Test(Attention_Gated):
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

            
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt


def gather_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    var_list = [torch.zeros_like(rt) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(var_list, rt, async_op=False)
    return [i.item() for i in var_list]


def eval_model(args, datasets, dataloader, model, phase):
    model.eval()
    all_labels = []
    all_values = []
    train_loss = 0
    
    prefetcher = data_prefetcher(dataloader)
    patches, label = prefetcher.next()
    index = 0
    while patches is not None:
        index += 1
        label = label[0].float()
        
        with torch.no_grad():
            Y_prob= model.forward(patches)
            Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

            J = -1. * (
                label * torch.log(Y_prob) + 
                (1. - label) * torch.log(1. - Y_prob)
            )
        
        reduced_loss = reduce_tensor(J.data)
        
        train_loss += reduced_loss.item()
        all_labels.extend(gather_tensor(label))
        all_values.extend(gather_tensor(Y_prob[0][0]))
        
        patches, label = prefetcher.next()
            
            
    if args.local_rank == 0:
        all_labels = np.array(all_labels)
        Loss = train_loss / len(all_labels)
        AUC, Acc = get_cm(all_labels, all_values)
        
        result = pd.DataFrame({
            'Pat': datasets.slide[:,0],
            'Slide': datasets.slide[:,1],
            'Label': all_labels,
            'Value': all_values
        })
        result.to_csv(f'./Results/Test_result.csv')

    return AUC, Loss

    
def get_cm(AllLabels, AllValues):
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
    print("[AUC/{:.4f}] [Threshold/{:.6f}] [Acc/{:.4f}]".format(Auc, t,  Acc))
    print("{:.2f}% {:.2f}%".format(cm[0][0]/ Neg_num * 100, cm[0][1]/Neg_num * 100))
    print("{:.2f}% {:.2f}%".format(cm[1][0]/ Pos_num * 100, cm[1][1]/Pos_num * 100))
    
    return Auc, Acc

def get_auc(ture, pred):
    fpr, tpr, thresholds = metrics.roc_curve(ture, pred, pos_label=1)
    return metrics.auc(fpr, tpr)



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method=args.init_method
    )
            
    
    with open(f'./Labels/test.json3') as f:
        test_data_map = json.load(f)

    ###
    test_slide_path = f'../pediatrics/Test_patches_normalized/'
    test_cut_ptid = os.listdir(test_slide_path)

    test_Label_df = pd.read_excel('./Labels/Test_data.xlsx')
    test_label_ptid = list(map(str, test_Label_df['ptid'].tolist()))

    ###
    test_data_avlb = []
    for sl in os.listdir(test_slide_path):
        if os.path.exists(f'{test_slide_path}/{sl}/{data_set}/{mag}'):
            if len(os.listdir(f'{test_slide_path}/{sl}/{data_set}/{mag}')) > 1:
                test_data_avlb.append(sl)

    test_label = list(set(test_label_ptid)&set(test_data_avlb))
    test_label.sort()


    ####
    device = torch.device(f"cuda:{args.local_rank}")

    model = apex.parallel.convert_syncbn_model(
        Attention_Gated(USEmodel, True)
    ).to(device)

    model = amp.initialize(model,opt_level="O0", keep_batchnorm_fp32=None)
    model = DistributedDataParallel(model, delay_allreduce=True)
    model.load_state_dict(torch.load(modelpath))

    ####
    test_datasets, test_loader = prepare_dataset(test_slide_path, mag)
    testauc, testloss = eval_model(args, test_datasets, test_loader, model, 'Test')
