import os
import h5py
import random
import numpy as np
import pdb
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_patch_util import *
from skimage.transform import resize
from scipy import ndimage, misc
import matplotlib.pyplot as plt


class PET_Train(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root

        self.data_dir = os.path.join(self.root, 'train')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.AUG = opts.AUG
        self.zoomsize_train = opts.zoomsize_train
        self.cropsize_train = opts.cropsize_train
        self.rotate_train = opts.rotate_train

    def __getitem__(self, index):
        data_file = self.data_files[index]

        with h5py.File(data_file, 'r') as f:
            vol_G1LD = f['G1LD'][...]
            vol_G2LD = f['G2LD'][...]
            vol_G3LD = f['G3LD'][...]
            vol_G4LD = f['G4LD'][...]
            vol_G5LD = f['G5LD'][...]
            vol_G6LD = f['G6LD'][...]
            vol_G1HD = f['G1HD'][...]
            vol_G2HD = f['G2HD'][...]
            vol_G3HD = f['G3HD'][...]
            vol_G4HD = f['G4HD'][...]
            vol_G5HD = f['G5HD'][...]
            vol_G6HD = f['G6HD'][...]

        if self.AUG:
            # zoom to size
            vol_G1LD = resize(vol_G1LD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G2LD = resize(vol_G2LD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G3LD = resize(vol_G3LD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G4LD = resize(vol_G4LD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G5LD = resize(vol_G5LD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G6LD = resize(vol_G6LD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G1HD = resize(vol_G1HD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G2HD = resize(vol_G2HD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G3HD = resize(vol_G3HD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G4HD = resize(vol_G4HD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G5HD = resize(vol_G5HD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
            vol_G6HD = resize(vol_G6HD, self.zoomsize_train, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)

            # random cropping
            dx = self.zoomsize_train[0] - self.cropsize_train[0]
            dy = self.zoomsize_train[1] - self.cropsize_train[1]
            dz = self.zoomsize_train[2] - self.cropsize_train[2]
            assert vol_G1LD.shape[0] >= self.cropsize_train[0]
            assert vol_G1LD.shape[1] >= self.cropsize_train[1]
            assert vol_G1LD.shape[2] >= self.cropsize_train[2]
            sx = random.randint(0, dx)
            sy = random.randint(0, dy)
            sz = random.randint(0, dz)

            vol_G1LD = vol_G1LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G2LD = vol_G2LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G3LD = vol_G3LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G4LD = vol_G4LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G5LD = vol_G5LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G6LD = vol_G6LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G1HD = vol_G1HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G2HD = vol_G2HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G3HD = vol_G3HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G4HD = vol_G4HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G5HD = vol_G5HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
            vol_G6HD = vol_G6HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]

            # random rotation
            ang = random.randint(-self.rotate_train, self.rotate_train)
            vol_G1LD = ndimage.rotate(vol_G1LD, ang, axes=(1, 2), reshape=False)
            vol_G2LD = ndimage.rotate(vol_G2LD, ang, axes=(1, 2), reshape=False)
            vol_G3LD = ndimage.rotate(vol_G3LD, ang, axes=(1, 2), reshape=False)
            vol_G4LD = ndimage.rotate(vol_G4LD, ang, axes=(1, 2), reshape=False)
            vol_G5LD = ndimage.rotate(vol_G5LD, ang, axes=(1, 2), reshape=False)
            vol_G6LD = ndimage.rotate(vol_G6LD, ang, axes=(1, 2), reshape=False)
            vol_G1HD = ndimage.rotate(vol_G1HD, ang, axes=(1, 2), reshape=False)
            vol_G2HD = ndimage.rotate(vol_G2HD, ang, axes=(1, 2), reshape=False)
            vol_G3HD = ndimage.rotate(vol_G3HD, ang, axes=(1, 2), reshape=False)
            vol_G4HD = ndimage.rotate(vol_G4HD, ang, axes=(1, 2), reshape=False)
            vol_G5HD = ndimage.rotate(vol_G5HD, ang, axes=(1, 2), reshape=False)
            vol_G6HD = ndimage.rotate(vol_G6HD, ang, axes=(1, 2), reshape=False)

        # covert to torch tensor
        vol_G1LD = torch.from_numpy(vol_G1LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G2LD = torch.from_numpy(vol_G2LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G3LD = torch.from_numpy(vol_G3LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G4LD = torch.from_numpy(vol_G4LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G5LD = torch.from_numpy(vol_G5LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G6LD = torch.from_numpy(vol_G6LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G1HD = torch.from_numpy(vol_G1HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G2HD = torch.from_numpy(vol_G2HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G3HD = torch.from_numpy(vol_G3HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G4HD = torch.from_numpy(vol_G4HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G5HD = torch.from_numpy(vol_G5HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G6HD = torch.from_numpy(vol_G6HD.copy()).unsqueeze(0).unsqueeze(0)
        vols_G12356LD = torch.cat([vol_G1LD, vol_G2LD, vol_G3LD, vol_G5LD, vol_G6LD], 0)
        vols_G12356HD = torch.cat([vol_G1HD, vol_G2HD, vol_G3HD, vol_G5HD, vol_G6HD], 0)
        vol_zeros = torch.zeros(vol_G1HD.shape)

        return {'vol_G1LD': vol_G1LD,
                'vol_G2LD': vol_G2LD,
                'vol_G3LD': vol_G3LD,
                'vol_G4LD': vol_G4LD,
                'vol_G5LD': vol_G5LD,
                'vol_G6LD': vol_G6LD,
                'vol_G1HD': vol_G1HD,
                'vol_G2HD': vol_G2HD,
                'vol_G3HD': vol_G3HD,
                'vol_G4HD': vol_G4HD,
                'vol_G5HD': vol_G5HD,
                'vol_G6HD': vol_G6HD,
                'vols_G12356LD': vols_G12356LD,
                'vols_G12356HD': vols_G12356HD,
                'vol_zeros': vol_zeros
                }

    def __len__(self):
        return len(self.data_files)


class PET_Test(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root

        self.data_dir = os.path.join(self.root, 'test')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.cropsize_train = opts.cropsize_train

    def __getitem__(self, index):
        data_file = self.data_files[index]

        with h5py.File(data_file, 'r') as f:
            vol_G1LD = f['G1LD'][...]
            vol_G2LD = f['G2LD'][...]
            vol_G3LD = f['G3LD'][...]
            vol_G4LD = f['G4LD'][...]
            vol_G5LD = f['G5LD'][...]
            vol_G6LD = f['G6LD'][...]
            vol_G1HD = f['G1HD'][...]
            vol_G2HD = f['G2HD'][...]
            vol_G3HD = f['G3HD'][...]
            vol_G4HD = f['G4HD'][...]
            vol_G5HD = f['G5HD'][...]
            vol_G6HD = f['G6HD'][...]

        sx = 0
        sy = 0
        sz = 0
        vol_G1LD = vol_G1LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G2LD = vol_G2LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G3LD = vol_G3LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G4LD = vol_G4LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G5LD = vol_G5LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G6LD = vol_G6LD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G1HD = vol_G1HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G2HD = vol_G2HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G3HD = vol_G3HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G4HD = vol_G4HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G5HD = vol_G5HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]
        vol_G6HD = vol_G6HD[sx:sx + self.cropsize_train[0], sy:sy + self.cropsize_train[1], sz:sz + self.cropsize_train[2]]

        # covert to torch tensor
        vol_G1LD = torch.from_numpy(vol_G1LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G2LD = torch.from_numpy(vol_G2LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G3LD = torch.from_numpy(vol_G3LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G4LD = torch.from_numpy(vol_G4LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G5LD = torch.from_numpy(vol_G5LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G6LD = torch.from_numpy(vol_G6LD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G1HD = torch.from_numpy(vol_G1HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G2HD = torch.from_numpy(vol_G2HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G3HD = torch.from_numpy(vol_G3HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G4HD = torch.from_numpy(vol_G4HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G5HD = torch.from_numpy(vol_G5HD.copy()).unsqueeze(0).unsqueeze(0)
        vol_G6HD = torch.from_numpy(vol_G6HD.copy()).unsqueeze(0).unsqueeze(0)
        vols_G12356LD = torch.cat([vol_G1LD, vol_G2LD, vol_G3LD, vol_G5LD, vol_G6LD], 0)
        vols_G12356HD = torch.cat([vol_G1HD, vol_G2HD, vol_G3HD, vol_G5HD, vol_G6HD], 0)
        vol_zeros = torch.zeros(vol_G1HD.shape)

        return {'vol_G1LD': vol_G1LD,
                'vol_G2LD': vol_G2LD,
                'vol_G3LD': vol_G3LD,
                'vol_G4LD': vol_G4LD,
                'vol_G5LD': vol_G5LD,
                'vol_G6LD': vol_G6LD,
                'vol_G1HD': vol_G1HD,
                'vol_G2HD': vol_G2HD,
                'vol_G3HD': vol_G3HD,
                'vol_G4HD': vol_G4HD,
                'vol_G5HD': vol_G5HD,
                'vol_G6HD': vol_G6HD,
                'vols_G12356LD': vols_G12356LD,
                'vols_G12356HD': vols_G12356HD,
                'vol_zeros': vol_zeros
                }

    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    pass
