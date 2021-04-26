python test.py \
--resume './outputs/experiment_train_svrhddn_dp_gan_mse_fold1/checkpoints/model_419.pt' \
--experiment_name 'experiment_test_svrhddn_dp_gan_mse_fold1' \
--model_type 'model_svrhd_dp_gan' \
--data_root '../Data/preprocess/Processed1_5Percent_nofilter_fold1/' \
--net_G1 'svr_dp' \
--net_G2 'unet' \
--net_D 'patchGAN'
