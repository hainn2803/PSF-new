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
from flow_model import Model

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


def get_dataset(dataroot, npoints,category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def train(gpu, opt, output_dir, noises_init):

    set_seed(opt)
    logger = setup_logging(output_dir)
    logger.info(f"Check if exist pretrained weight: {os.path.exists(opt.model)}")
    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.workers = 0

        opt.saveEpoch =  int(opt.saveEpoch / opt.ngpus_per_node)
        opt.diagEpoch = int(opt.diagEpoch / opt.ngpus_per_node)
        opt.vizEpoch = int(opt.vizEpoch / opt.ngpus_per_node)

    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, None)

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)


    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

    if opt.model != '':
        ckpt = torch.load(opt.model, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])

    start_epoch = 0
    
    print(start_epoch, opt.nEpochs)

    for epoch in range(start_epoch, opt.nEpochs):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        for i, data in enumerate(dataloader):
            x = data['train_points'].transpose(1,2)
            noises_batch = noises_init[data['idx']].transpose(1,2)

            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                x = x.cuda(gpu)
                noises_batch = noises_batch.cuda(gpu)
            elif opt.distribution_type == 'single':
                x = x.cuda()
                noises_batch = noises_batch.cuda()
            break

        logger.info('Generation: eval')

        model.eval()
        with torch.no_grad():

            x0 = []
            x1 = []
            for i in range(200):
                x_gen_eval = model.gen_samples((64, *x.shape[1:]), x.device, clip_denoised=False)
                x0.append(x_gen_eval[0])
                x1.append(x_gen_eval[1])

                print(f"x_gen_eval[0].shape: {x_gen_eval[0].shape}, x_gen_eval[1].shape: {x_gen_eval[1].shape}")

                print('Step', i)

            x0 = torch.cat(x0, 0)
            x1 = torch.cat(x1, 0)
            # This should be parallely sampled with 8 GPUs in practice , here we only use one as example
            print(f"x0.shape: {x0.shape}, x1.shape: {x1.shape}")
            torch.save([x0, x1], 'DATASET.pth')

            exit(0)

    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        dist.destroy_process_group()

def main():
    opt = parse_args()
    if 1:
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)

    ''' workaround '''
    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category)
    noises_init = torch.randn(len(train_dataset), opt.npoints, opt.num_channels)

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        train(opt.gpu, opt, output_dir, noises_init)



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data/ShapeNetCore.v2.PC15k/')
    parser.add_argument('--category', default='chair')

    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--nEpochs', type=int, default=20000, help='number of epochs to train for')

    parser.add_argument('--num_channels', default=3)
    parser.add_argument('--npoints', default=2048)

    '''model'''
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    '''params'''
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')
    parser.add_argument('--model', default='./data/epoch_8431.pth', help="path to model (to sample)")

    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default=None, choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveEpoch', default=500, help='unit: epoch')
    parser.add_argument('--diagEpoch', default=500, help='unit: epoch')
    parser.add_argument('--vizEpoch', default=500, help='unit: epoch')
    parser.add_argument('--printFreqIter', default=50, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    opt = parser.parse_args()

    return opt

def print_gpu_info():
    if not torch.cuda.is_available():
        print("No GPU available")
    else:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


if __name__ == '__main__':
    print_gpu_info()
    main()