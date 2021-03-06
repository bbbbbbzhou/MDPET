CUDA_VISIBLE_DEVICES=1 python train.py \
--experiment_name 'experiment_train_svrhddn_dp_gan_mse' \
--model_type 'model_svrhd_dp_gan' \
--data_root './preprocess/' \
--net_G1 'svr_dp' \
--net_G2 'unet' \
--net_D 'patchGAN' \
--weight_dn_recon 100 \
--weight_reg 1000 \
--image_loss 'mse' \
--batch_size 1 \
--zoomsize_train 106 106 106 \
--cropsize_train 96 96 96 \
--rotate_train 30 \
--AUG \
--eval_epochs 20 \
--save_epochs 20 \
--snapshot_epochs 20 \
--lr 1e-4
