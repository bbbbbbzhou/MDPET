import os
import argparse
import json
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from utils import prepare_sub_folder
from datasets import get_datasets
from models import create_model
import scipy.io as sio
import csv

parser = argparse.ArgumentParser(description='MDPET')

# model name
parser.add_argument('--experiment_name', type=str, default='experiment_train_svrhddn_dpa_mse', help='give a experiment name before training')
parser.add_argument('--model_type', type=str, default='model_svrhd_dpa', help='give a model name before training: model_svrhd / model_svrld / model_vm')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
parser.add_argument('--data_root', type=str, default='../Data/preprocess/Processed1_5Percent_nofilter/', help='data root folder')
parser.add_argument('--dataset', type=str, default='PET', help='dataset name')

# network architectures (registration)
parser.add_argument('--net_G1', type=str, default='svr_dpa', help='generator network for registration: svr / vr / svr_dpa / vr_dpa')
parser.add_argument('--int_steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int_downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', default=False, action='store_true', help='enable bidirectional cost function')
# if u-net
parser.add_argument('--enc1', type=int, nargs='+', default=[16, 32, 32, 32], help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec1', type=int, nargs='+', default=[32, 32, 32, 32, 16], help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
# if dpa-net
parser.add_argument('--enc1_1', type=int, nargs='+', default=[16, 32, 32, 32], help='fix/mov vol encoder for denoising')
parser.add_argument('--dec1_1', type=int, nargs='+', default=[32, 32, 32, 32], help='fix/mov vol decoder for denoising')
parser.add_argument('--dec1_2', type=int, nargs='+', default=[32, 32, 32, 32, 32, 16], help='decoder for motion estimation')

# network architectures (final denoising)
parser.add_argument('--net_G2', type=str, default='unet', help='generator network for denoising: unet')
parser.add_argument('--enc2', type=int, nargs='+', default=[32, 64, 128, 256], help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec2', type=int, nargs='+', default=[256, 256, 128, 64, 32, 1], help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--net_D', type=str, default='patchGAN', help='discriminator')

# overall loss options
parser.add_argument('--weight_dn_recon', type=float, default=100, help='weight of denoising recon loss')
parser.add_argument('--weight_reg', type=float, default=1000, help='weight of registration loss')

# register loss options
parser.add_argument('--image_loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--weight_deform', type=float, default=0.01, help='weight of deformation loss (default: 0.01)')

# training options
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--zoomsize_train', nargs='+', type=int, default=[106, 106, 106], help='randomly crop patch size for train')
parser.add_argument('--cropsize_train', nargs='+', type=int, default=[96, 96, 96], help='randomly crop patch size for train')
parser.add_argument('--rotate_train', type=int, default=20, help='randomly rotate patch along z for train')
parser.add_argument('--AUG', default=False, action='store_true', help='use augmentation')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=2, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=2, help='save evaluation for every number of epochs')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=2, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=100, help='save model for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# optimizer
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=1000, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay ratio for step scheduler')

# other
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(opts)
model.setgpu(opts.gpu_ids)

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters: {} \n'.format(num_param))

if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume)

model.set_scheduler(opts, ep0)
ep0 += 1
print('Start training at epoch {} \n'.format(ep0))

# select dataset
train_set, val_set = get_datasets(opts)
train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

with open(os.path.join(output_directory, 'options.json'), 'w') as f:
    f.write(options_str)

with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(model.loss_names)

# training loop
for epoch in range(ep0, opts.n_epochs + 1):

    train_bar = tqdm(train_loader)
    model.train()
    model.set_epoch(epoch)
    for it, data in enumerate(train_bar):
        total_iter += 1
        model.set_input(data)
        model.optimize()
        train_bar.set_description(desc='[Epoch {}]'.format(epoch) + model.loss_summary)

        if it % opts.log_freq == 0:
            with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(model.get_current_losses().values())

    model.update_learning_rate()

    # save checkpoint
    if (epoch+1) % opts.snapshot_epochs == 0:
        print('Saving checkpoint ......')
        checkpoint_name = os.path.join(checkpoint_directory, 'model_{}.pt'.format(epoch))
        model.save(checkpoint_name, epoch, total_iter)

    # evaluation
    if (epoch+1) % opts.eval_epochs == 0:
        print('Evaluation ......')

        model.eval()
        with torch.no_grad():
            model.evaluate(val_loader)

        with open(os.path.join(output_directory, 'metrics_table.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,
                             model.mi_Gmean,
                             model.mi_G1, model.mi_G2, model.mi_G3, model.mi_G5, model.mi_G6,
                             model.mse_DN])

    if (epoch+1) % opts.save_epochs == 0:
        sio.savemat(os.path.join(image_directory, 'eval.mat'), model.results)
