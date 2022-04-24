import sys
import os

sys.path.append(os.getcwd())

import json
import yaml
from tqdm import tqdm
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader
from src.utils import load_logger, load_yaml
from src.sampler import get_data_sampler, load_datasets
from src.lr_schedule import WarmupLinearScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from distutils.util import strtobool as _bool
from src.train_utils import get_data_feeder_from_sampler
from models.distance_func import *
from models.loss_func import *
from src.trainer import *
from copy import deepcopy

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--file_dir', type=str, default='/mnt/data/user8/solar-commonsense_inference')
parser.add_argument('--dataset_type', type=str, default='atomic-2020')#, required=True)
parser.add_argument('--model_name', type=str, default='gpt2', help="bart | gpt2")#, required=True)
parser.add_argument('--model_size', type=str, default='base', help="base | large")#, required=True)
parser.add_argument('--main_yml', type=str, default='v01.yml')#, required=True)
parser.add_argument('--tknz_yml', type=str, default='tokenizer_config.yml')
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('--log', type=str, default='NEWvTEST')#, required=True)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--load_model', type=str, default=None)

# Optimize
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--update_batch', type=int, default=128)
parser.add_argument('--iter_per_epoch', type=int, default=40000)
parser.add_argument('--dev_iter_per_epoch', type=int, default=1)
parser.add_argument('--epoch_num', type=int, default=1)

parser.add_argument("--learning_rate", default=0.00001, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--lm_head_dropout_p', default=0.1, type=float)
parser.add_argument('--proj_head_dropout_p', default=0.1, type=float)

parser.add_argument("--recadam_anneal_fun", type=str, default='sigmoid', choices=["sigmoid", "linear", 'constant'],
                    help="the type of annealing function in RecAdam. Default sigmoid")
parser.add_argument("--recadam_anneal_k", type=float, default=0.0001, help="k for the annealing function in RecAdam.")
parser.add_argument("--recadam_anneal_t0", type=int, default=5000, help="t0 for the annealing function in RecAdam.")
parser.add_argument("--recadam_anneal_w", type=float, default=1.0,
                    help="Weight for the annealing function in RecAdam. Default 1.0.")
parser.add_argument("--recadam_pretrain_cof", type=float, default=5000.0,
                    help="Coefficient of the quadratic penalty in RecAdam. Default 5000.0.")
parser.add_argument('--warmup_steps', type=int, default=100)

# ETC
parser.add_argument('--share_proj_layer', type=_bool, default=True, help=
                    'if true, share projection layer between s_representation and ro_representation'
                    'if false, do not share projection layer between them. each have each own projection layer')

args = parser.parse_args()

if args.batch_size > args.update_batch:
    args.batch_size = args.update_batch

# Path Setting
np.random.seed(args.random_seed)

# - log, [tb, ckpt, gen, eval]
main_config_path = f'config/{args.dataset_type}/{args.model_name}-{args.model_size}_experiments/{args.main_yml}'
tknz_config_path = f'config/{args.tknz_yml}'
dataset_config_path = f'config/{args.dataset_type}/datasets.yml'

log_dir = f'{args.file_dir}/log/{args.dataset_type}/{args.model_name}-{args.model_size}_experiments/{args.log}'
logging_dir = log_dir + '/logging.log'
tb_dir = log_dir + '/tb'
gen_dir = log_dir + '/gen'
eval_dir = log_dir + '/eval'
ckpt_dir = log_dir + '/ckpt'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    os.mkdir(tb_dir)
    os.mkdir(gen_dir)
    os.mkdir(eval_dir)
    os.mkdir(ckpt_dir)

# Initialize Log
logger = load_logger(logging_dir, args.log_level)
logger.info('Logger is Successfully initialized !')
tb_writer = SummaryWriter(tb_dir)

# Loading YML file
main_config = load_yaml(main_config_path)
tknz_config = load_yaml(tknz_config_path)
dataset_cfg = load_yaml(dataset_config_path)

with open(os.path.join(log_dir, 'main_config.json'), 'w') as f:
    json.dump(main_config, f)
with open(os.path.join(log_dir, 'tknz_config.json'), 'w') as f:
    json.dump(tknz_config, f)
with open(os.path.join(log_dir, 'argparse.json'), 'w') as f:
    json.dump(args.__dict__, f)

task_cfg = main_config['task']
model_cfg = main_config['model']
opt_cfg = main_config['opt']
log_cfg = main_config['log']

# Tokenizer Setting with args.tknz_yml
if args.model_name == 'bart':
    from transformers import BartTokenizer as Tokenizer
    from models.bart import CometBART as CometModel
    from src.tokenizer import BartCSKGTokenizer as CSKGTokenizer

elif args.model_name == 'gpt2':
    from transformers import GPT2Tokenizer as Tokenizer
    from models.gpt2 import CometGPT2 as CometModel
    from src.tokenizer import Gpt2CSKGTokenizer as CSKGTokenizer

elif 't5' in model_cfg['name']:
    raise NotImplementedError
else:
    raise NotImplementedError

anneal_targets = load_yaml(f'models/pretrained_params/pretrained_{args.model_name}-{args.model_size}.yml')['pretrained_weights']

_tokenizer = Tokenizer.from_pretrained(model_cfg['pretrained_model'])
tokenizer = CSKGTokenizer(_tokenizer, tknz_config)
vocab_len = len(tokenizer) + 1

model = CometModel.from_pretrained(model_cfg['pretrained_model'])
model.modify_lm_heads(vocab_len, args.lm_head_dropout_p)
model.add_proj_layer(model_cfg['contrastive_head'], args.proj_head_dropout_p)

pretrained_model = deepcopy(model)

# Load Dataset
dataset, sim_mat = load_datasets(dataset_cfg, tokenizer, logger)

# Initialize Sampler
sampler = get_data_sampler(task_cfg, dataset, sim_mat, tokenizer, logger, args)
train_sampler = sampler['train']
dev_sampler = sampler['dev']

# Connect sampler to data_feeding adaptor
options = main_config['model']['task_adaptor_options']

iter_num = 0
assert args.update_batch >= args.batch_size
stack_num = int(args.update_batch / args.batch_size)
assert int(stack_num * args.batch_size) == int(args.update_batch)

train_data_feeder = get_data_feeder_from_sampler(train_sampler, options, tokenizer, task_cfg, args.batch_size, args.iter_per_epoch * stack_num)
dev_data_feeder = get_data_feeder_from_sampler(dev_sampler, options, tokenizer, task_cfg, args.batch_size, args.dev_iter_per_epoch)

if model_cfg['contrastive_head']['proj_layer_type'] in ['multi-head', 'multi-head-deeper']:
    distance_model = MultiHeadDistance(model_cfg['contrastive_head']['multi-head'])
elif model_cfg['contrastive_head']['proj_layer_type'] == 'non-linear':
    distance_model = NonLinearHeadDistance()
else:
    raise NotImplementedError

usable_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if usable_cuda else "cpu")

model.to(device)
pretrained_model.to(device)

from src.rec_adam_wrapper import get_adam_optimizer
optim, scheduler = get_adam_optimizer(model, pretrained_model, anneal_targets, args, args.iter_per_epoch * args.epoch_num)

global_steps = 0
model.train()
optim.zero_grad()

if args.share_proj_layer:
    share_s = share_ro = None
else:
    share_s = True
    share_ro = False

share_proj_layer = (share_s, share_ro)

if task_cfg['con_task']['loss_func'] == 'NT-Logistic':
    con_loss_func = NT_Logistic(opt_cfg['temperature'])
else:
    raise NotImplementedError

auxil_loss_func = torch.nn.CrossEntropyLoss(reduction='none')


con_task_loader = DataLoader(train_data_feeder['con_task'], batch_size=1,
                             drop_last=False, shuffle=False, num_workers=4)

if options['con_task']['format'] == 'enc-dec':
    con_task_train = ConTaskTrainerForEncDec(model, device, distance_model, con_loss_func,
                            log_cfg['tb_period'], tb_writer, share_proj_layer, args)
    
elif options['con_task']['format'] == 'dec':
    con_task_train = ConTaskTrainerForDec(model, device, distance_model, con_loss_func,
                            log_cfg['tb_period'], tb_writer, share_proj_layer, args)

if 'rec_inf_shu_task' in train_data_feeder:
    auxil_task_loader = DataLoader(train_data_feeder['rec_inf_shu_task'], batch_size=args.batch_size,
                                   drop_last=True, shuffle=False, num_workers=10)
    auxil_task_name = 'rec_inf_shu_task'
    auxil_task_train = RecTaskTrainerForEncDec(model, device, auxil_loss_func, log_cfg['tb_period'], tb_writer)

elif 'mask_gen_task' in train_data_feeder:
    auxil_task_loader = DataLoader(train_data_feeder['mask_gen_task'], batch_size=args.batch_size,
                                   drop_last=True, shuffle=False, num_workers=10)
    auxil_task_name = 'mask_gen_task'
    auxil_task_train = MaskGenTaskTrainerForDec(model, device, auxil_loss_func, log_cfg['tb_period'], tb_writer)

else:
    raise NotImplementedError

torch.save(model, os.path.join(ckpt_dir, 'model-{}-steps.ckpt'.format(global_steps)))

#########
# for task_name, task_feeder in train_data_feeder.items():
#     task_feeder.sampler._init_cursor(args.random_seed)
#
# for con_task_data, auxil_task_data in tqdm(zip(con_task_loader, auxil_task_loader)):
#     multi_task_loss = []
########
for e in range(args.epoch_num):
    model.train()
    for task_name, task_feeder in train_data_feeder.items():
        task_feeder.sampler._init_cursor(args.random_seed)

    for con_task_data, auxil_task_data in tqdm(zip(con_task_loader, auxil_task_loader)):
        multi_task_loss = []

        with autocast():
            con_task_data = {i: con_task_data[i][0] for i in con_task_data}
            con_loss = con_task_train.train(con_task_data, global_steps, iter_num == 0)
            multi_task_loss.append((con_loss.item(), task_cfg['con_task']['weight']))
            con_loss *= task_cfg['con_task']['weight']
            con_loss /= stack_num
        con_loss.backward()

        with autocast():
            auxil_loss = auxil_task_train.train(auxil_task_data, global_steps, iter_num == 0)
            multi_task_loss.append((auxil_loss.item(), task_cfg[auxil_task_name]['weight']))
            auxil_loss *= task_cfg[auxil_task_name]['weight']
            auxil_loss /= stack_num
        auxil_loss.backward()

        iter_num += 1
        if iter_num != stack_num:
            continue
        iter_num = 0

        clip_grad_norm_(model.parameters(), opt_cfg['clip_grad_norm'])
        _, anneal_lambda = optim.step()
        scheduler.step()
        global_steps += 1

        if global_steps % log_cfg['tb_period'] == 0:
            avg_multi_task_loss = sum([i[0] for i in multi_task_loss]) / len(multi_task_loss)
            weighted_multi_task_loss = sum([i[0] * i[1] for i in multi_task_loss])
            tb_writer.add_scalar('train/total_lr', optim.param_groups[0]['lr'], global_steps)
            tb_writer.add_scalar('train/multi-task_loss', avg_multi_task_loss, global_steps)
            tb_writer.add_scalar('train/multi-task_loss(weighted)', weighted_multi_task_loss, global_steps)
            tb_writer.add_scalar('train/anneal_lambda', anneal_lambda, global_steps)
            tb_writer.flush()

        if global_steps % log_cfg['save_period'] == 0:
            torch.save(model, os.path.join(ckpt_dir, 'model-{}-steps.ckpt'.format(global_steps)))

torch.save(model, os.path.join(ckpt_dir, 'model-{}-steps.ckpt'.format(global_steps)))

