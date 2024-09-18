from datasets import ClusterDataset
from models import *
import torch.nn.functional as F
import torch
import os
import utils
import numpy as np
import shutil
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import visualization
import random
import umap
from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE as tsne
import matplotlib.colors as col
import matplotlib.cm as cm

import warnings
warnings.filterwarnings("ignore")


########################################################
def run(eval_dataloader, device, dim_redu, random_state, P, data_set, dim, mod):

    zs, zc_logit, zc = [], [], []
    y_true = []

    with torch.no_grad():
        for data in eval_dataloader:
            x, y_true_ = data

            x = x.to(device, non_blocking=True)

            zs_, zc_logit_, _ = encoder(x)
            zc_ = F.softmax(zc_logit_, dim=1)

            zs.append(zs_.detach().cpu().numpy())
            zc.append(zc_.detach().cpu().numpy())
            zc_logit.append(zc_logit_.detach().cpu().numpy())
            y_true.append(y_true_.detach().cpu().numpy())

    zs = np.concatenate(zs, axis=0)
    zc = np.concatenate(zc, axis=0)
    zc_logit = np.concatenate(zc_logit, axis=0)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.argmax(zc_logit, axis=1)

    #######################
    show_num = 1000000
    match = utils.hungarian_match(y_pred, y_true, dim)
    y_pred = utils.convert_cluster_assignment_to_ground_truth(y_pred, match)
    z = np.concatenate([zs, zc], axis=1)
#     patches_path = np.array(d.dataset.imgs)

#     patches_path_1 = list(patches_path[:,0])
    
#     print(patches_path_1)
#     patches_path_1 = [0 if f.split('|')[0].split('/')[-1]=='NEG' else 1 for f in patches_path_1]
#     print(patches_path_1)
    
    ##############
    Z = []
#     X = []
    Y = []

    for cluster in range(dim):
        c_idxs = (y_pred == cluster)
        try:
            z_new = z[c_idxs]
            if len(z_new) > show_num:
                top_idxs = np.random.choice(len(z_new), show_num)
                Z.append(z_new[top_idxs])
#                 X.append(np.array(patches_path_1)[c_idxs][top_idxs])
                Y.append([cluster]*len(z_new[top_idxs]))
            else:
                top_idxs = np.arange(0, len(z_new), 1)
                Z.append(z_new)
#                 X.append(np.array(patches_path_1)[c_idxs])
                Y.append([cluster]*len(z_new))
#             if os.path.exists('{}/{}'.format(P, cluster)):
#                 shutil.rmtree('{}/{}'.format(P, cluster))
#             os.makedirs('{}/{}'.format(P, cluster))
#             pos = 0
#             for path, _ in patches_path[c_idxs][top_idxs]:
#                 shutil.copy(path, 
#                            '{}/{}/{}'.format(P, cluster, os.path.basename(path)))
#                 if path.split('/')[4].split('|')[0]=='POS':
#                     pos += 1
            print('cluster {} number: {}'.format(cluster+1, sum(c_idxs)))
        except Exception:
            print('ERROR')
            pass

    Z = np.concatenate(Z)
#     X = np.concatenate(X)
    Y = np.concatenate(Y)
    
#     print(X)
    
    ##########################    
    if dim_redu == 'umap':
        reducer = umap.UMAP(random_state=random_state, n_neighbors=500)
        embedding = reducer.fit_transform(Z)
    else:
        reducer = tsne(n_iter=20000, random_state=random_state)
        embedding = reducer.fit_transform(Z)
        
    T = []
    for i in Y:
        i = int(i)
        T.append(f'cluster-{i+1}')

    # sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.spines['bottom'].set_linewidth(1.)
    ax.spines['left'].set_linewidth(1.)

    ter = sns.scatterplot(x=embedding[:,0], 
                          y=embedding[:,1], 
                          hue=T, 
    #                       ec='white', 
                          alpha=.8,
                          s=30,
                          ax=ax)

    handles, labels = ter.get_legend_handles_labels()
    ter.legend(fontsize=5,
               ncol=1, loc='upper right',
               frameon=True)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    sns.despine()
    
    fig.savefig(f'{P}/{data_set}_{dim}_cluster_{dim_redu}_{mod}.png')
    print(f'{P}/{data_set}_{dim}_cluster_{dim_redu}_{mod}.png -- has been saved')
    
    return
    



if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
    data_set = 'HE_20_6'
    dim = 6
    dim_redu = 'tsne'
    np.random.seed(999)
    random.seed(999)
    random_state=99999999   

    d = ClusterDataset(f'./Cluster_patches_copy_0821/HE_20_top_patches', data_set, training=False)
    eval_dataloader = DataLoader(d, batch_size=128, 
                                 num_workers=16, 
                                 pin_memory=True, 
                                 shuffle=False,
                                 drop_last=False)
    print('patch number:', len(d))
    
    P = f'./Cluster_patches_dim_0826/{data_set}_top_patches/'
    if os.path.exists(P):
        pass
    else:
        os.mkdir(P)
    
    for step in range(149, 3000, 50):
        encoder = get_encoder(50, dim)
        critic = get_critic(50, dim)
        discriminator = get_discriminator(50, dim)

        device = torch.device('cuda:0')
        encoder = nn.DataParallel(encoder)
        critic = nn.DataParallel(critic)
        discriminator = nn.DataParallel(discriminator)

        mod = f'encoder_{step}'
        print(mod)
        utils.load_model(encoder, f'./checkpoint_copy_0826/{data_set}/DCCS/model/{mod}.tar')    # 4

        encoder.to(device)
        encoder.eval()

        run(eval_dataloader, device, dim_redu, random_state, P, data_set, dim, mod)
        