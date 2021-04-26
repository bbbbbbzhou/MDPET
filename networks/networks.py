import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from .convolutional_rnn import Conv3dLSTM
from . import layers
from .modelio import LoadableModel, store_config_args


class TimeDistributed(nn.Module):
    # modified by Luyao Dec.12, 2019
    # input data size is not limited to 1-D anymore, but can be N-D.
    # Assuming the first 2 dimensions are sample number and time steps.
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)
        input_size = x.size()[2:]
        input_size = torch.tensor(input_size).numpy()
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view((-1,) + tuple(input_size))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)

        # We have to reshape Y
        output_size = y.size()[1:]
        output_size = torch.tensor(output_size).numpy()
        if self.batch_first:
            y = y.contiguous().view((x.size(0),) + (-1,) + tuple(output_size))  # (samples, timesteps, output_size)
        else:
            y = y.view((-1,) + (x.size(1),) + tuple(output_size))  # (timesteps, samples, output_size)

        return y


class TimeDistributed2(nn.Module):
    # modified by Luyao Dec.12, 2019
    # input data size is not limited to 1-D anymore, but can be N-D.
    # Assuming the first 2 dimensions are sample number and time steps.
    def __init__(self, module, batch_first=False):
        super(TimeDistributed2, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x1, x2):

        if len(x1.size()) <= 2:
            return self.module(x1)
        input1_size = x1.size()[2:]
        input1_size = torch.tensor(input1_size).numpy()
        # Squash samples and timesteps into a single axis
        x1_reshape = x1.contiguous().view((-1,) + tuple(input1_size))  # (samples * timesteps, input_size)

        if len(x2.size()) <= 2:
            return self.module(x2)
        input2_size = x2.size()[2:]
        input2_size = torch.tensor(input2_size).numpy()
        # Squash samples and timesteps into a single axis
        x2_reshape = x2.contiguous().view((-1,) + tuple(input2_size))  # (samples * timesteps, input_size)

        y = self.module(x1_reshape, x2_reshape)

        # We have to reshape Y
        output_size = y.size()[1:]
        output_size = torch.tensor(output_size).numpy()
        if self.batch_first:
            y = y.contiguous().view((x1.size(0),) + (-1,) + tuple(output_size))  # (samples, timesteps, output_size)
        else:
            y = y.view((-1,) + (x1.size(1),) + tuple(output_size))  # (timesteps, samples, output_size)

        return y

##########################################
# Sequential Voxel Register
##########################################
class UConvLSTMnet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, use_lstm=True):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = TimeDistributed(nn.Upsample(scale_factor=2, mode='nearest'), batch_first=True)

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(TimeDistributed(ConvBlock(ndims, prev_nf, nf, stride=2), batch_first=True))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(TimeDistributed(ConvBlock(ndims, channels, nf, stride=1), batch_first=True))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(TimeDistributed(ConvBlock(ndims, prev_nf, nf, stride=1), batch_first=True))
            prev_nf = nf

        # configure 3d conv lstm
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.convlstm3d1 = Conv3dLSTM(in_channels=16,  # Corresponds to input size
                                          out_channels=16,  # Corresponds to hidden size
                                          kernel_size=3,  # Int or List[int]
                                          num_layers=1,
                                          bidirectional=True,
                                          dilation=1, stride=1, dropout=0.5,
                                          batch_first=True)

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=2)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        # final conv lstm (if bidirectional -> channel size double)
        if self.use_lstm:
            x, h = self.convlstm3d1(x)

        return x


class SVR(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_lstm=True):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = UConvLSTMnet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            use_lstm=use_lstm
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        if use_lstm:
            self.flow = Conv(self.unet_model.dec_nf[-1] * 2, ndims, kernel_size=3, padding=1)  # x2 -> bidirectional lstm
        else:
            self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.flow = TimeDistributed(self.flow, batch_first=True)

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = TimeDistributed(layers.ResizeTransform(int_downsize, ndims), batch_first=True) if resize else None
        self.fullsize = TimeDistributed(layers.ResizeTransform(1 / int_downsize, ndims), batch_first=True) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = TimeDistributed(layers.VecInt(down_shape, int_steps), batch_first=True) if int_steps > 0 else None

        # configure transformer
        self.transformer = TimeDistributed2(layers.SpatialTransformer(inshape), batch_first=True)

        # configure bidirectional training
        self.bidir = bidir

    def forward(self,
                x_LD_src, x_LD_tgt,
                x_HD_src, x_HD_tgt):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x_LD = torch.cat([x_LD_src, x_LD_tgt.repeat(1, 5, 1, 1, 1, 1)], 2)
        x = self.unet_model(x_LD)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_LD_src = self.transformer(x_LD_src, pos_flow)
        y_LD_tgt = self.transformer(x_LD_tgt, neg_flow) if self.bidir else None

        y_HD_src = self.transformer(x_HD_src, pos_flow)
        y_HD_tgt = self.transformer(x_HD_tgt, neg_flow) if self.bidir else None

        # return non-integrated flow field (preint_folw) / integrated flow field (pos_flow)
        if self.bidir:
            return y_HD_src, y_HD_tgt, y_LD_src, y_LD_tgt, preint_flow, pos_flow
        else:
            return y_HD_src, y_LD_src, preint_flow, pos_flow


##########################################
# Sequential Voxel Register with Dual-Stream Pyramid Attention
##########################################
class DPConvLSTMnet(nn.Module):
    """
    A dual-stream pyramid attention architecture (two unet + one decoder). Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder1_1 / encoder1_2: [16, 32, 32, 32]
        decoder1_1 / decoder1_2: [32, 32, 32, 16, 1]
        decoder1_3: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, use_lstm=True):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # build feature list automatically
        self.enc_nf_dn, self.dec_nf_dn, self.dec_nf_motion = nb_features

        self.upsample = TimeDistributed(nn.Upsample(scale_factor=2, mode='nearest'), batch_first=True)

        # configure denoising encoder (down-sampling path)
        prev_nf_dn = 1
        self.downarm_dn1 = nn.ModuleList()
        self.downarm_dn2 = nn.ModuleList()
        for nf_dn in self.enc_nf_dn:
            self.downarm_dn1.append(TimeDistributed(ConvBlock(ndims, prev_nf_dn, nf_dn, stride=2), batch_first=True))
            self.downarm_dn2.append(TimeDistributed(ConvBlock(ndims, prev_nf_dn, nf_dn, stride=2), batch_first=True))
            prev_nf_dn = nf_dn

        # configure denoising decoder (up-sampling path)
        enc_dn_history = list(reversed(self.enc_nf_dn))
        self.uparm_dn1 = nn.ModuleList()
        self.uparm_dn2 = nn.ModuleList()
        for i, nf_dn in enumerate(self.dec_nf_dn[:len(self.enc_nf_dn)]):
            channels = prev_nf_dn + enc_dn_history[i] if i > 0 else prev_nf_dn
            self.uparm_dn1.append(TimeDistributed(ConvBlock(ndims, channels, nf_dn, stride=1), batch_first=True))
            self.uparm_dn2.append(TimeDistributed(ConvBlock(ndims, channels, nf_dn, stride=1), batch_first=True))
            prev_nf_dn = nf_dn
        self.lastconv_dn1 = TimeDistributed(nn.Conv3d(prev_nf_dn + 1, 1, 1, 1), batch_first=True)
        self.lastconv_dn2 = TimeDistributed(nn.Conv3d(prev_nf_dn + 1, 1, 1, 1), batch_first=True)

        # configure motion decoder (up-sampling path)
        dec_dn_history = list(self.dec_nf_dn)
        self.uparm_motion = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf_motion[:len(self.dec_nf_dn)]):
            channels = dec_dn_history[i]
            self.uparm_motion.append(TimeDistributed(ConvBlock(ndims, channels, nf, stride=1), batch_first=True))
            prev_nf = nf

        # configure extra motion decoder convolutions (no up-sampling)
        self.extras = nn.ModuleList()
        for nf in self.dec_nf_motion[len(self.enc_nf_dn):]:
            self.extras.append(TimeDistributed(ConvBlock(ndims, prev_nf, nf, stride=1), batch_first=True))
            prev_nf = nf

        # configure 3d conv lstm
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.convlstm3d1 = Conv3dLSTM(in_channels=16,  # Corresponds to input size
                                          out_channels=16,  # Corresponds to hidden size
                                          kernel_size=3,  # Int or List[int]
                                          num_layers=1,
                                          bidirectional=True,
                                          dilation=1, stride=1, dropout=0.5,
                                          batch_first=True)

    def forward(self, x_mov, x_fix):

        # get encoder activations --- denosing
        x_enc_dn1 = [x_mov]
        for layer in self.downarm_dn1:
            x_enc_dn1.append(layer(x_enc_dn1[-1]))

        x_enc_dn2 = [x_fix]
        for layer in self.downarm_dn2:
            x_enc_dn2.append(layer(x_enc_dn2[-1]))

        # conv, upsample, concatenate series --- denosing
        x_dn1 = x_enc_dn1.pop()
        x_dec_dn1_motion = []
        for layer in self.uparm_dn1:
            x_dn1 = layer(x_dn1)
            x_dec_dn1_motion.append(x_dn1)
            x_dn1 = self.upsample(x_dn1)
            x_dn1 = torch.cat([x_dn1, x_enc_dn1.pop()], dim=2)
        x_dn1 = self.lastconv_dn1(x_dn1)

        x_dn2 = x_enc_dn2.pop()
        x_dec_dn2_motion = []
        for layer in self.uparm_dn2:
            x_dn2 = layer(x_dn2)
            x_dec_dn2_motion.append(x_dn2)
            x_dn2 = self.upsample(x_dn2)
            x_dn2 = torch.cat([x_dn2, x_enc_dn2.pop()], dim=2)
        x_dn2 = self.lastconv_dn2(x_dn2)

        # conv, upsample, concatenate series --- motion decoder
        x_dec_dn1_motion.reverse()
        x_dec_dn2_motion.reverse()
        x_motion = x_dec_dn1_motion.pop() + x_dec_dn2_motion.pop()
        for i, layer in enumerate(self.uparm_motion):
            x_motion = layer(x_motion)
            x_motion = self.upsample(x_motion)
            if i < len(self.uparm_motion)-1:
                x_motion = x_motion + x_dec_dn1_motion.pop() + x_dec_dn2_motion.pop()

        # extra convs at full resolution -- motion decoder
        for layer in self.extras:
            x_motion = layer(x_motion)

        # final conv lstm (if bidirectional -> channel size double)
        if self.use_lstm:
            x_motion, h = self.convlstm3d1(x_motion)

        return x_motion, x_dn1, x_dn2


class SVR_DP(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_dpnet_features=None,
        nb_dpnet_levels=None,
        dpnet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_lstm=True):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.dpnet_model = DPConvLSTMnet(
            inshape,
            nb_features=nb_dpnet_features,
            nb_levels=nb_dpnet_levels,
            feat_mult=dpnet_feat_mult,
            use_lstm=use_lstm
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        if use_lstm:
            self.flow = Conv(self.dpnet_model.dec_nf_motion[-1] * 2, ndims, kernel_size=3, padding=1)  # x2 -> bidirectional lstm
        else:
            self.flow = Conv(self.dpnet_model.dec_nf_motion[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.flow = TimeDistributed(self.flow, batch_first=True)

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = TimeDistributed(layers.ResizeTransform(int_downsize, ndims), batch_first=True) if resize else None
        self.fullsize = TimeDistributed(layers.ResizeTransform(1 / int_downsize, ndims), batch_first=True) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = TimeDistributed(layers.VecInt(down_shape, int_steps), batch_first=True) if int_steps > 0 else None

        # configure transformer
        self.transformer = TimeDistributed2(layers.SpatialTransformer(inshape), batch_first=True)

        # configure bidirectional training
        self.bidir = bidir

    def forward(self,
                x_LD_src, x_LD_tgt,
                x_HD_src, x_HD_tgt):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x_motion, x_HD_src_pred, x_HD_tgt_pred = self.dpnet_model(x_LD_src, x_LD_tgt.repeat(1, 5, 1, 1, 1, 1))

        # transform into flow field
        flow_field = self.flow(x_motion)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_LD_src = self.transformer(x_LD_src, pos_flow)
        y_LD_tgt = self.transformer(x_LD_tgt, neg_flow) if self.bidir else None

        y_HD_src = self.transformer(x_HD_src, pos_flow)
        y_HD_tgt = self.transformer(x_HD_tgt, neg_flow) if self.bidir else None

        # return non-integrated flow field (preint_folw) / integrated flow field (pos_flow)
        if self.bidir:
            return y_HD_src, y_HD_tgt, \
                   y_LD_src, y_LD_tgt, \
                   preint_flow, pos_flow, \
                   x_HD_src_pred, x_HD_tgt_pred
        else:
            return y_HD_src, \
                   y_LD_src, \
                   preint_flow, pos_flow, \
                   x_HD_src_pred, x_HD_tgt_pred


##########################################
# VoxelMorph #
##########################################
class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 1
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock_DN(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock_DN(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 1
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ConvBlock_DN(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        Norm = getattr(nn, 'InstanceNorm%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.norm = Norm(out_channels, affine=False)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, norm_layer='IN'):
        super(Discriminator, self).__init__()

        nf = 64
        model = []
        model += [LeakyReLUConv3d(in_channels, nf, kernel_size=4, stride=2, padding=1)]
        model += [LeakyReLUConv3d(nf, nf * 2, kernel_size=4, stride=2, padding=1, norm=norm_layer)]
        model += [LeakyReLUConv3d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, norm=norm_layer)]
        model += [LeakyReLUConv3d(nf * 4, nf * 8, kernel_size=4, stride=1, norm=norm_layer)]
        model += [nn.Conv3d(nf * 8, 1, kernel_size=1, stride=1, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)

        return out


class Dis(nn.Module):
    def __init__(self, input_dim, n_layer=3, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64
        self.model = self._make_net(input_dim, ch, n_layer, norm, sn)

    def _make_net(self, input_dim, ch, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv3d(input_dim, ch, kernel_size=4, stride=2, padding=1, norm='None', sn=sn)]
        tch = ch
        for i in range(n_layer-1):
            model += [LeakyReLUConv3d(tch, tch * 2, kernel_size=4, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        if sn:
            pass
        else:
            model += [nn.Conv3d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


##################################################################################
# Basic Blocks
##################################################################################
class LeakyReLUConv3d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv3d, self).__init__()
        model = []
        model += [nn.ReplicationPad3d(padding)]
        if sn:
            pass
        else:
            model += [nn.Conv3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        if norm == 'IN':
            model += [nn.InstanceNorm3d(n_out, affine=False)]
        elif norm == 'BN':
            model += [nn.BatchNorm3d(n_out)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    pass

