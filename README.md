# MDPET: A Unified Motion Correction and Denoising Adversarial Network for Low-dose Gated PET

Bo Zhou, Yu-Jung Tsai, Xiongchao Chen, James S. Duncan and Chi Liu

IEEE Transactions on Medical Imaging (TMI), 2021

[[Paper](https://www.xxx)]

This repository contains the PyTorch implementation of MDPET.

### Citation
If you use this code for your research or project, please cite:

    @article{zhou2021mdpet,
      title={MDPET: A Unified Motion Correction and Denoising Adversarial Network for Low-dose Gated PET},
      author={Zhou, Bo and Tsai, Yu-Jung and Chen, Xiongchao and Duncan, James S and Liu, Chi},
      journal={IEEE Transactions on Medical Imaging},
      year={2021},
      publisher={IEEE}
    }


### Environment and Dependencies
Requirements:
* Python 3.7
* Pytorch 1.4.0
* scipy
* scikit-image
* itertools
* tqdm
* csv
* json

Our code has been tested with Python 3.7, Pytorch 1.4.0, CUDA 10.0 on Ubuntu 18.04.


### Dataset Setup
    ./preprocess/              # data setup 
    │
    ├── train                  # training data -- each .h5 contains 6 x high-dose gated data and 6 x low-dose gated data
    │   ├── case_1.h5     
    │   ├── case_2.h5       
    │   ├── case_3.h5 
    │   ├── ...    
    │   └── case_N.h5
    │
    ├── test                   # test data -- each .h5 contains 6 x high-dose gated data and 6 x low-dose gated data
    │   ├── case_N+1.h5     
    │   ├── case_N+2.h5       
    │   ├── case_N+3.h5 
    │   ├── ...    
    │   └── case_M.h5         
    └── 
Each .h5 file should contain 6 gated image at high-dose level and 6 gated image at low-dose level.  
The variable name in the .h5 should be 'G1LD' / 'G2LD' / 'G3LD' / 'G4LD' / 'G5LD' / 'G6LD' for low-dose gated images, and 'G1HD' / 'G2HD' / 'G3HD' / 'G4HD' / 'G5HD' / 'G6HD' for high-dose gated images.


### To Run Our Code
- Train the model
```bash
python train.py \
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
```
where \
`--experiment_name` provides the experiment name for the current run, and save all the corresponding results under the experiment_name's folder. \
`--data_root` provides the data folder directory (with structure and files illustrated above). \
`--net_G1` specifies the motion estimation network type. \
`--net_G2` specifies the denoising network type. \
`--net_D` specifies the discriminator network type. \
`--AUG` enables the data augumentation during the training. \
Other hyperparameters can be adjusted in the code as well.

- Test the model
```bash
python test.py \
--resume './outputs/experiment_train_svrhddn_dp_gan_mse/checkpoints/model_xxx.pt' \
--experiment_name 'experiment_test_svrhddn_dp_gan_mse' \
--model_type 'model_svrhd_dp_gan' \
--data_root './preprocess/' \
--net_G1 'svr_dp' \
--net_G2 'unet' \
--net_D 'patchGAN'
```
Sample training/test scripts are provided under './scripts/' and can be directly executed.


### Test Data
We provide one test data which includes the input to MDPET and output of the MDPET.  
Please see the following link to request the test data.  
[[Drive Link](https://drive.google.com/drive/folders/1w8AlHien6GeRGJ1xpVxyobFZfUuOWv2z?usp=sharing)]


### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```