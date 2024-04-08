import wandb

from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter

def main():
    os.putenv('CUDA_VISIBLE_DEVICES', '0')
    if not torch.cuda.is_available():
        print("No GPU found")
        exit(1)
    # else:
    #     print("GPU found")
    #     exit(0)
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)
    config.max_epoch = args.max_epoch
    config.total_bs = args.total_bs
    config.model.num_query = args.num_queries
    config.optimizer.type = args.opt
    config.optimizer.kwargs.lr = args.opt_lr
    config.optimizer.kwargs.weight_decay = args.opt_wd
    config.optimizer.lambda_sparse_dense = args.opt_lambda_sparse_dense
    config.scheduler.type = args.sched
    config.bnmscheduler.bn_decay = args.bnmsched_decay
    config.bnmscheduler.bn_momentum = args.bnmsched_momentum
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 

    # run
    if args.test:
        test_net(args, config)
    else:
        wandb.init(
            project='MT',
            name=args.exp_name,
            tags=['AdaPoinTr'],
            config={**args.__dict__, **config}
        )
        wandb.define_metric('train/*', step_metric='epoch')
        wandb.define_metric('val/*', step_metric='epoch')
        run_net(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
