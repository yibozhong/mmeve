for H_LR in 2e-4 2e-5 2e-6; do
    for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele; do
        CUDA_VISIBLE_DEVICES=0 python train.py --dataset $DATASET --h_lr $H_LR --qwen --mm_trained
    done
done


# for H_LR in 2e-4 2e-5 2e-6; do
#     for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele; do
#         CUDA_VISIBLE_DEVICES=0 python train.py --dataset $DATASET --h_lr $H_LR --mm_trained
#     done
# done