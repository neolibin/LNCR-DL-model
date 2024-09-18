export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--checkpoint_root='./checkpoint_copy_0826' \
--dataset_type='HE_20_6' \
--img_type='rgb' \
--dataset_path='./Cluster_patches_copy_0821/HE_20_top_patches' \
--beta_aug=2 \
--batch_size=128 \
--dim_zc=6
    
# ps aux | grep train | awk '{print "kill -9 " $2}'| sh