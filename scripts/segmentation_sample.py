"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import matplotlib.pyplot as plt
import nibabel as nib
from visdom import Visdom
#viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.cardiacloader import CardiacDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.nn import mean_flat
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    colour_map = np.array([[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255],[255,255,255]])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    #ds = BRATSDataset(args.data_dir, test_flag=True)
    ds = CardiacDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.to(th.device(f"cuda"))
    model = DistributedDataParallel(model, device_ids=[0])
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location=th.device('cuda'))
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    p = 0
    while len(all_images) * args.batch_size < args.num_samples:

        p += 1

        for i in range(16):
            b, m, path = next(data)  #should return an image from the dataloader "data"
        
        # b, m, path = next(data)  #should return an image from the dataloader "data"

        m_start = th.clone(m)
    
        c = th.randn_like(m)

        img = th.cat((b, c), dim=1)     #add a noise channel$
     
        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        temp = m.numpy()
        np.putmask(temp, temp == -1 , 0)
        m = th.tensor(temp)

        arg_max_mask = (th.argmax(m, dim=1) + 1).cpu().numpy()
        val_max_mask = th.max(m,dim=1)[0].cpu().numpy()
        np.putmask(arg_max_mask, val_max_mask == 0, 0)
        m = th.Tensor(arg_max_mask)
  

        rgb_m = colour_map[arg_max_mask.squeeze()].transpose().transpose(1,2,0)
 
        plt.imshow(rgb_m)
        plt.axis('off')
        plt.savefig(os.path.join("cardiac_64_samples", f"mask_{p}.png"), bbox_inches="tight", pad_inches=0)


        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
       
            arg_max_sample = (th.argmax(sample, dim=1) + 1).cpu().numpy()
            val_max_sample = th.max(sample,dim=1)[0].cpu().numpy()
            np.putmask(arg_max_sample, val_max_sample < 0.5, 0)
     
            rgb_sample = colour_map[arg_max_sample.squeeze()].transpose()

            plt.imshow(rgb_sample.transpose(1,2,0))
            plt.axis('off')
            plt.savefig(os.path.join("cardiac_64_samples", f"sample_{p}_{i}.png"), bbox_inches="tight", pad_inches=0)


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
