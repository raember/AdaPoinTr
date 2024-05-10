import os
import argparse
from pathlib import Path

import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)   
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true',
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=50, help = 'test freq')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')
    parser.add_argument('--total_bs', type=int, default=32, help='BS [SWEEP]')
    parser.add_argument('--max_epoch', type=int, default=2000, help='epochs [SWEEP]')
    parser.add_argument('--num_queries', type=int, default=512, help='num_queries [SWEEP]')
    parser.add_argument('--opt', type=str, default='AdamW', help='optimizer [SWEEP]')
    parser.add_argument('--opt_lr', type=float, default=0.00001, help='lr (optimizer) [SWEEP]')
    parser.add_argument('--opt_wd', type=float, default=0.00003, help='weight decay (optimizer) [SWEEP]')
    parser.add_argument('--lambda_sparse_dense', type=float, default=0.5, help='lambda sparse vs dense loss (optimizer) [SWEEP]')
    # parser.add_argument('--opt_warmingup_e', type=int, default=0, help='warm up epochs (optimizer) [SWEEP]')
    parser.add_argument('--sched', type=str, default='LambdaLR', help='scheduler [SWEEP]')
    parser.add_argument('--sched_lrd', type=float, default=0.9, help='lr decay (scheduler) [SWEEP]')
    parser.add_argument('--bnmsched_decay', type=float, default=0.5, help='bn decay (BNM scheduler) [SWEEP]')
    parser.add_argument('--bnmsched_momentum', type=float, default=0.9, help='bn momentum (BNM scheduler) [SWEEP]')
    parser.add_argument('--n_points', type=int, default=10000, help='Number of input points')
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    args.exp_name += f"_{np.random.randint(0, 1e5)}"
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path, exist_ok=True)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

