import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_generation import PVCNN2Base
import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds

class Flowmodel:
    def __init__(self, opt):
        self.num_timesteps = opt.time_num
        return

    def p_mean(self, denoise_fn, data, t):
        model_output = denoise_fn(data, t)
        model_mean = data + model_output * 1. / self.num_timesteps
        return model_mean


    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean = self.p_mean(denoise_fn, data=data, t=t)

        return model_mean


    def p_sample_loop(self, denoise_fn, shape, device,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in range(self.num_timesteps):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in range(self.num_timesteps):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    @torch.no_grad()
    def sample_pairs(self, data, x0 = None):
        if x0 is None:
            x0 = torch.randn_like(data)

        z0 = x0
        t= torch.rand((data.shape[0], 1, 1)).to(data.device)
        inter_data = t * data + (1.-t) * z0
        target = data - z0
        return inter_data, t * 999, target

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """
        Training loss calculation
        """
        inter_data, t, target = self.sample_pairs(data_start)
        t = t.squeeze()
        data_t = inter_data
        eps_recon = denoise_fn(data_t, t)
        losses = ((target - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))

        return losses

class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )


class Model(nn.Module):
    def __init__(self, args, betas):
        super(Model, self).__init__()
        self.flow = Flowmodel(args)

        self.model = PVCNN2(num_classes=args.num_channels, embed_dim=args.embed_dim, use_att=args.attention,
                            dropout=args.dropout, extra_feature_channels=0)


    def _denoise(self, data, t):
        B, D,N= data.shape
        assert data.dtype == torch.float

        out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None):

        B, D, N = data.shape
        t = torch.randint(0, 1000, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.flow.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.flow.p_sample_loop(self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.flow.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

