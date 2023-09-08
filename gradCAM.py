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
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import cv2
import time
import glob
import argparse
from PIL import Image
import apex
from apex import amp
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel
from apex.multi_tensor_apply import multi_tensor_applier
from torch.utils.data import DataLoader as DL
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader as DL, Sampler
import json
import pandas as pd
import random
import torchvision
from thop import profile


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--slide_path', default='None', type=str)
    parser.add_argument('--modelpath', default='None', type=str)
    parser.add_argument('--group', default='None', type=str)
    parser.add_argument('--model', default=18, type=str)
    parser.add_argument('--batchsize', default='64', type=str)
    parser.add_argument('--data_set', default='HE', type=str)
    parser.add_argument('--mag', default='5', type=str)
    parser.add_argument('--patchlimit', default=50, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()


args = get_parser()


img_transform = transforms.Compose([
#             transforms.CenterCrop(384),
            transforms.Resize(299)
        ])


class PedDataset(Dataset):
    def __init__(self, Data_path, ptid, slide, Mag='5', transforms=None, limit=96):
        self.ptids = ptid
        self.slide = [(ptid, slide)]
        
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
        
    
class TestDistSlideSampler(DistributedSampler):
    def __init__(self, k, dataset, limit=512):
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
#         shuffle = torch.randperm(
#             self.limit if patch_num > self.limit else patch_num
#         ).tolist()
#         return  indice[shuffle]
        if patch_num > self.limit:
            indice = indice[k*self.limit:min(patch_num, (k+1)*self.limit)]
            return np.array(indice).flatten()
        else:
            return np.array(indice).flatten()
    

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

    
def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def get_model(path, USEmodel, device):
    model = apex.parallel.convert_syncbn_model(
        Attention_Gated(USEmodel, True)
    ).to(device)
    model = amp.initialize(model,opt_level="O0", keep_batchnorm_fp32=None)
    model = DistributedDataParallel(model, delay_allreduce=True)
    model.load_state_dict(torch.load(path))
    return model

if __name__ == '__main__':
    args = get_parser()
    
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method=args.init_method
    )
    
    device = torch.device(f"cuda:{args.local_rank}")
    model = get_model(model_path, USEmodel, device)
    
    
    slide_path = f'../pediatrics/Combine_patches_normalized/'
    save_path = f'./gradCAM/'

    with open(f'./Combine_Labels/Combine_pat_labels_treat12_ALL.json3') as f:
        data_map = json.load(f)

    ###
    Model_Label_df = pd.read_excel('./Combine_Labels/FAH_Pats_split20221014.xlsx')
    train_label_ptid = list(map(str, Model_Label_df[Model_Label_df['group']==0]['ptid'].tolist()))
 
    ###
    Model_data_avlb = []
    for sl in os.listdir(slide_path):
        if os.path.exists(f'{slide_path}/{sl}/{data_set}/{mag}'):
            if len(os.listdir(f'{slide_path}/{sl}/{data_set}/{mag}')) > 1:
                Model_data_avlb.append(sl)

    cam_label = list(set(train_label_ptid)&set(Model_data_avlb))
    cam_label.sort()


    for vl in cam_label:
        try:
            cam_datasets = PedDataset(slide_path, 
                                       vl, 
                                       data_set,
                                       Mag=mag, 
                                       transforms=transforms.Compose([
                                            transforms.Resize(299),
                                        ]),
                                       limit=2)

            memory_format = torch.contiguous_format
            collate_fn = lambda b: fast_collate(b, memory_format)

            numslide = len(os.listdir(slide_path+'/'+vl+'/'+data_set+'/'+mag+'/'))
            times = numslide//int(PatchLimit)+1

            for k in range(times):
                print('Sampling patch:', k*int(PatchLimit), 'to', min(numslide, (k+1)*int(PatchLimit)))
                sampler = TestDistSlideSampler(k, cam_datasets, limit=PatchLimit)
                cam_loader = DL(cam_datasets, 
                                 batch_sampler=sampler,
                                 num_workers=16,
                                 pin_memory=True,
                                 collate_fn=collate_fn)

                layer_name = get_last_conv_name(model.module.feature_extractor)
                model.eval()

                prefetcher = data_prefetcher(cam_loader)
                patches, label = prefetcher.next()
                index = 0

                if os.path.exists(save_path):
                    if os.path.exists(f'{save_path}/CAM_{group}'):
                        pass
                    else:
                        os.makedirs(f'{save_path}/CAM_{group}')
                else:
                    os.makedirs(save_path)
                    os.makedirs(f'{save_path}/CAM_{group}')

                while patches is not None:
                    ptid, slide = sampler.slide[index]
                    if os.path.exists(f'{save_path}/CAM_{group}/{ptid}'):
                        if os.path.exists(f'{save_path}/CAM_{group}/{ptid}/{slide}/{mag}'):
                            pass
                        else:
                            os.makedirs(f'{save_path}/CAM_{group}/{ptid}/{slide}/{mag}')
                    else:
                        os.makedirs(f'{save_path}/CAM_{group}/{ptid}')
                        os.makedirs(f'{save_path}/CAM_{group}/{ptid}/{slide}/{mag}')
                    idxs = sampler.get_slide(ptid, slide)
                    path = [cam_datasets.patch[i] for i in idxs]

                    model.zero_grad()

                    handler = []
                    feature = None
                    gradient = None

                    def get_feature_hook(module, input, output):
                        global feature
                        feature = output

                    def get_grads_hook(module, input, output):
                        global gradient
                        gradient = output[0]

                    for (name, module) in \
                        model.module.feature_extractor.named_modules():
                        if name == layer_name:
                            handler.append(module.register_forward_hook(get_feature_hook))
                            handler.append(module.register_backward_hook(get_grads_hook))


                    Y_prob = model.forward(patches)
                    Y_prob.backward()
            
                    for i in range(len(idxs)):
                        f = feature[i].cpu().data.numpy() # 256 * 8 * 8
                        g = gradient[i].cpu().data.numpy() # 256 * 8 * 8
                        weight = np.mean(g, axis=(1, 2)) # 256, 

                        cam = f * weight[:, np.newaxis, np.newaxis] # 256 * 8 * 8
                        cam = np.sum(cam, axis=0) # 256, 
                        cam -= np.min(cam)
                        cam /= np.max(cam)
                        cam = cv2.resize(cam, (299, 299))

                        img = Image.open(path[i])
                        img = img_transform(img)
                        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

                        heatmap = cam*0.6 + img * 0.4
                        heatmap = np.vstack((heatmap, img))

                        file_name = '{}_CAM.jpeg'.format(os.path.basename(path[i]).split('.')[0])
                        cv2.imwrite(f'{save_path}/CAM_{group}/{ptid}/{slide}/{mag}/{file_name}', heatmap)

                    for h in handler:
                        h.remove()
                    patches, label = prefetcher.next()
                    index += 1
        except Exception:
            print(f'No such file or directory: {slide_path}/{vl}/{data_set}/{mag}/')