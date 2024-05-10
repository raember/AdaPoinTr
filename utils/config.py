import numpy as np
import yaml
from easydict import EasyDict
import os
from .logger import print_log

def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger = logger)

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger = logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger = logger)

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)        
    return config

def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger = logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger = logger)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    n_points = args.n_points
    pow = np.log2(n_points / 625)  # 5**4
    n_out = 2 ** (10 + int(pow))
    for key, ds in config.dataset.items():
        config['dataset'][key]._base_.FROM_N = n_points
        config['dataset'][key]._base_.TO_N = n_out
        config['dataset'][key].FROM_N = n_points
        config['dataset'][key].TO_N = n_out
    config.model.num_points = n_out
    config.max_epoch = args.max_epoch
    config.total_bs = args.total_bs
    config.model.num_query = args.num_queries
    config.optimizer.type = args.opt
    config.optimizer.kwargs.lr = args.opt_lr
    config.optimizer.kwargs.weight_decay = args.opt_wd
    config.lambda_sparse_dense = args.lambda_sparse_dense
    config.scheduler.type = args.sched
    config.bnmscheduler.bn_decay = args.bnmsched_decay
    config.bnmscheduler.bn_momentum = args.bnmsched_momentum
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    return config

def save_experiment_config(args, config, logger = None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))
    print_log(f'Copy the Config file from {args.config} to {config_path}',logger = logger )