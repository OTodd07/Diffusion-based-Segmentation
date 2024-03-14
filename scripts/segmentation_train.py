"""
Train a diffusion model on images.
"""
import sys
import argparse
sys.path.append("..")
sys.path.append(".")
import json
from guided_diffusion import dist_util, logger, ddp_utils
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.cardiacloader import CardiacDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

import torch as th
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)

def main():
    args = create_argparser().parse_args()
    with open("configs/config.json", 'r') as f:
        config = json.load(f)

    # print(config)
    # print(args)

    # print(7/0)

    #dist_util.setup_dist()
    device, rank, world_size = ddp_utils.setup_distributed()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    kwargs = {'batch_size': args.batch_size, 'num_workers': 1, 'pin_memory': True}
    # model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("creating data loader...")
    #ds = BRATSDataset(args.data_dir, test_flag=False)
    ds = CardiacDataset(args.data_dir, test_flag=False)
    datal= th.utils.data.DataLoader(
        ds,
        sampler=DistributedSampler(ds, shuffle=True),
        **kwargs)
    data = iter(datal)
    print(device)
    th.cuda.set_device(device)
    model.to(th.device(f"cuda"))
    model = DistributedDataParallel(model, device_ids=[device])
    


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="./data/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
