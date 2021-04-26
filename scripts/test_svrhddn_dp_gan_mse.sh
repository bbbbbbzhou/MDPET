python test.py \
--resume './outputs/experiment_train_svrhddn_dp_gan_mse/checkpoints/model_xxx.pt' \
--experiment_name 'experiment_test_svrhddn_dp_gan_mse' \
--model_type 'model_svrhd_dp_gan' \
--data_root './preprocess/' \
--net_G1 'svr_dp' \
--net_G2 'unet' \
--net_D 'patchGAN'
