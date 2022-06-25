import sys, os

sys.path.append(os.getcwd())

import json
import yaml
from tqdm import tqdm
import argparse
import numpy as np
import torch
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from distutils.util import strtobool as _bool
from torch.utils.data import DataLoader

from src.utils import load_logger, load_yaml

from src.finetune.finetune_utils import *
from src.finetune.finetune_trainer import get_finetune_trainer
from models.model_utils import convert_model
from src.lr_schedule import WarmupLinearScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--file_dir', type=str, default='/mnt/data/user8/solar-commonsense_inference')
parser.add_argument('--dataset_type', type=str, default='conceptnet')
parser.add_argument('--model_name', type=str, default='bart', help="bart | gpt2")#, required=True)
parser.add_argument('--model_size', type=str, default='large', help="base | large")#, required=True)
parser.add_argument('--exp_type', type=str, default='baseline', help='baseline | experiments')#, required=True)
parser.add_argument('--main_yml', type=str, default='v01.yml')
parser.add_argument('--tknz_yml', type=str, default='tokenizer_config.yml')
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('--log', type=str, default='vTEST1')#, required=True)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--load_model', type=str, default=None)

parser.add_argument('--same_hwang', type=_bool, default=False)

parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--update_batch', type=int, default=64)
parser.add_argument('--epoch_num', type=int, default=2)
parser.add_argument('--output_drop', type=float, default=0.1)
parser.add_argument('--remove_except_best', type=_bool, default=True)
parser.add_argument('--patience', type=int, default=2)


args = parser.parse_args()

if args.load_model is not None:
    assert args.exp_type == 'experiments'

args.batch_size = args.batch_size if args.update_batch >= args.batch_size else args.update_batch

np.random.seed(args.random_seed)

main_config_path = f'config/{args.dataset_type}/{args.model_name}-{args.model_size}_baseline/{args.main_yml}'
tknz_config_path = f'config/{args.tknz_yml}'

dataset_config_path = f'config/{args.dataset_type}/datasets.yml'

log_dir = f'{args.file_dir}/log_fntn/{args.dataset_type}/{args.model_name}-{args.model_size}_{args.exp_type}/{args.log}'

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

logger = load_logger(logging_dir, args.log_level)
logger.info('Logger is Successfully initialized !')
tb_writer = SummaryWriter(tb_dir)

main_config = load_yaml(main_config_path)
tknz_config = load_yaml(tknz_config_path)
dataset_cfg = load_yaml(dataset_config_path)

with open(os.path.join(log_dir, 'main_config.json'), 'w') as f:
    json.dump(main_config, f)
with open(os.path.join(log_dir, 'tknz_config.json'), 'w') as f:
    json.dump(tknz_config, f)
with open(os.path.join(log_dir, 'argparse.json'), 'w') as f:
    json.dump(args.__dict__, f)

model_cfg = main_config['model']
opt_cfg = main_config['opt']
log_cfg = main_config['log']

# Model Selection
MODEL_TYPE = None
if 'bart' in model_cfg['name']:
    from transformers import BartTokenizer as Tokenizer
    from models.bart import CometBART as CometModel
    from src.tokenizer import BartCSKGTokenizer as CSKGTokenizer
    from models.bart import convert_BARTModel_to_BartForConditionalGeneration as model2gen
    MODEL_TYPE = 'enc-dec'

elif 't5' in model_cfg['name']:
    raise NotImplementedError

elif 'gpt2' in model_cfg['name']:
    from transformers import GPT2Tokenizer as Tokenizer
    from models.gpt2 import CometGPT2 as CometModel
    from src.tokenizer import Gpt2CSKGTokenizer as CSKGTokenizer
    from models.gpt2 import convert_GPT2Model_to_GPT2LMHeadModel as model2gen
    MODEL_TYPE = 'dec'

else:
    raise NotImplementedError

_tokenizer = Tokenizer.from_pretrained(model_cfg['pretrained_model'])
tokenizer = CSKGTokenizer(_tokenizer, tknz_config)

vocab_len = len(tokenizer) + 1

if args.load_model is None:
    model = CometModel.from_pretrained(model_cfg['pretrained_model'])
    model.modify_lm_heads(vocab_len, args.output_drop)
    logger.info("Model is loaded from : {}".format(model_cfg['pretrained_model']))
else:
    args.load_model = os.path.join(args.load_model)
    src_model = torch.load(args.load_model)
    logger.info("Model is loaded from : {}".format(args.load_model))
    dst_model = CometModel.from_pretrained(model_cfg['pretrained_model'])
    dst_model.modify_lm_heads(vocab_len, args.output_drop)
    model = convert_model(src_model, dst_model)

dataset = load_fntn_datasets(dataset_cfg, tokenizer, logger)

fntn_train_dataset = get_finetune_dataset(dataset['train'], dataset_cfg, tokenizer, logger, 'train', MODEL_TYPE)
fntn_dev_dataset = get_finetune_dataset(dataset['dev'], dataset_cfg, tokenizer, logger, 'dev', MODEL_TYPE)

eval_dev_dataset = get_eval_dataset(dataset['dev'], tokenizer, logger, 'dev', MODEL_TYPE)
eval_test_dataset = get_eval_dataset(dataset['test'], tokenizer, logger, 'test', MODEL_TYPE)

fntn_train_loader = DataLoader(fntn_train_dataset,
                               batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=20)
fntn_dev_loader = DataLoader(fntn_dev_dataset,
                             batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=20)

usable_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if usable_cuda else "cpu")
model.to(device)

# Optimizer Settings
if args.same_hwang is not None:
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
else:
    optim = torch.optim.Adam(model.parameters(),
             lr=args.lr, betas=(opt_cfg['adam_beta_1'], opt_cfg['adam_beta_2']), weight_decay=opt_cfg['weight_decay'])

total_steps = int((len(fntn_train_dataset) / args.update_batch) * args.epoch_num) + 100
lr_schedule = WarmupLinearScheduler(optim, args.lr, opt_cfg['warmup_steps'], total_steps)
loss_func = torch.nn.CrossEntropyLoss(reduction='none')
global_step = 0
scaler = GradScaler()
model.train()
optim.zero_grad()

iter_num = 0

assert args.update_batch >= args.batch_size
stack_num = int(args.update_batch / args.batch_size)
assert int(stack_num * args.batch_size) == int(args.update_batch)

fntn_trainer = get_finetune_trainer(model, device, loss_func, log_cfg['tb_period'], tb_writer, MODEL_TYPE)

ckpt_loss = {}

for e in range(1, args.epoch_num+1):
    model.train()
    fntn_train_loader.dataset.shuffle()
    patience = args.patience
    for sample in tqdm(fntn_train_loader, desc='[Train] Epoch {}/{}'.format(e, args.epoch_num), ncols=130):
        with autocast():
            loss = fntn_trainer.train(sample, global_step, 'train', iter_num == 0)
            loss /= stack_num
            loss.backward()
        iter_num += 1
        if iter_num != stack_num:
            continue
        iter_num = 0

        lr_schedule(global_step)
        clip_grad_norm_(model.parameters(), opt_cfg['clip_grad_norm'])
        optim.step()
        optim.zero_grad()
        global_step += 1

        if global_step % log_cfg['tb_period'] == 0:
            tb_writer.add_scalar('train/lr', optim.param_groups[0]['lr'], global_step)
            tb_writer.flush()

    model.eval()
    loss_list = list()
    for sample in tqdm(fntn_dev_loader, desc='[Validate]', ncols=130):
        with autocast():
            loss = fntn_trainer.train(sample, global_step, 'dev')
            loss_list.append(loss.item())
    loss = sum(loss_list) / len(loss_list)
    tb_writer.add_scalar('dev/loss', loss, global_step)

    save_name = os.path.join(ckpt_dir, 'model-{}-epoch.ckpt'.format(e))

    torch.save(model, save_name)
    ckpt_loss[save_name] = loss

    for _ckpt, _loss in ckpt_loss.items():
        if loss > _loss:
            patience -= 1
    if patience <= 0:
        logger.info('Patience is over. End')
        break

tokenizer_save_name = os.path.join(ckpt_dir, 'tokenizer.torch-pkl')
torch.save(tokenizer, tokenizer_save_name)

with open(os.path.join(ckpt_dir, 'ckpt_loss.json'), 'w') as f:
    json.dump(ckpt_loss, f)

with open(os.path.join(ckpt_dir, 'ckpt_loss.json'), 'r') as f:
    ckpt_loss = json.load(f)

best_ckpt = None
best_loss = 10000

for _ckpt, _loss in ckpt_loss.items():
    if best_loss > _loss:
        best_loss = _loss
        best_ckpt = _ckpt

if args.remove_except_best:
    for _ckpt in ckpt_loss:
        if _ckpt == best_ckpt:
            continue
        os.remove(os.path.join(_ckpt))

decode = tokenizer.tokenizer.decode

model = torch.load(best_ckpt).to('cpu')
if MODEL_TYPE == 'enc-dec':
    gen_model = model2gen(model, model_cfg['pretrained_model']).to(device)
else:
    gen_model = model.to(device)

test_decode_results = {
    'info': {'log': log_dir, 'ckpt': best_ckpt},
    'content': list()}

greedy_test_decode_results = deepcopy(test_decode_results)
greedy_test_decode_results['info']['decode_method'] = 'greedy'
beam5_test_decode_results = deepcopy(test_decode_results)
beam5_test_decode_results['info']['decode_method'] = 'beam5'
nucl_test_decode_results = deepcopy(test_decode_results)
nucl_test_decode_results['info']['decode_method'] = 'nucl'

if MODEL_TYPE == 'enc-dec':
    for sample in tqdm(eval_test_dataset, ncols=130):
        src = sample['src']
        refs = sample['ref']
        enc_input_ids = torch.tensor(src).to(device).view(1, -1)
        enc_att_masks = torch.ones_like(enc_input_ids).to(device)

        inputs = {'input_ids': enc_input_ids, 'attention_mask': enc_att_masks}
        greedy_output = gen_model.generate(**inputs, early_stopping=True,
                                           bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
        # try:
        #     beam5_output = gen_model.generate(**inputs, num_beams=5,  early_stopping=True) #max_length=60,
        # except:
        #     beam5_output = gen_model.generate(**inputs, num_beams=5, early_stopping=True, max_length=60)
        greedy_str = decode(greedy_output.tolist()[0])
        # beam5_str = decode(beam5_output.tolist()[0])

        if '<gen>' in greedy_str:
            greedy_str = greedy_str[greedy_str.find('<gen>') + 1 + len('<gen>'):].strip()
        # if '<gen>' in beam5_str:
        #     beam5_str = beam5_str[beam5_str.find('<gen>')+1 + len('<gen>'):].strip()

        _input = decode(src)
        _refs = list()
        for ref in refs:
            _ref = decode(ref)
            _refs.append(_ref)

        greedy_test_decode_results['content'].append({'input': _input, 'output': greedy_str, 'refs': _refs})
        # beam5_test_decode_results['content'].append({'input': _input, 'output': beam5_str, 'refs': _refs})

    with open(os.path.join(log_dir, 'greedy_gen_examples.json'), 'w') as f:
        json.dump(greedy_test_decode_results, f)
    #
    # with open(os.path.join(log_dir, 'beam5_gen_examples.json'), 'w') as f:
    #     json.dump(beam5_test_decode_results, f)

else:

    for sample in tqdm(eval_test_dataset, ncols=130):
        src = sample['src']
        refs = sample['ref']
        enc_input_ids = torch.tensor(src).to(device).view(1, -1)
        enc_att_masks = torch.ones_like(enc_input_ids).to(device)

        inputs = {'input_ids': enc_input_ids, 'att_masks': enc_att_masks}

        for i in range(30):
            output = model.forward_conditional_gen(**inputs)  # ['input_ids'], att_masks=inputs['attention_mask'])
            gen_token_id = int(torch.argmax(output[:, -1, :], -1))
            old_inputs = {key: val.tolist() for key, val in inputs.items()}
            old_inputs['input_ids'][0].append(gen_token_id)
            old_inputs['att_masks'][0].append(1)
            if gen_token_id == tokenizer.eos_token_id:
                break
            inputs = {key: torch.tensor(val).to(device) for key, val in old_inputs.items()}

        greedy_output = inputs['input_ids']

        greedy_str = decode(greedy_output.tolist()[0])

        if '<gen>' in greedy_str:
            greedy_str = greedy_str[greedy_str.find('<gen>') + 1 + len('<gen>'):].strip()

        _input = decode(src)
        print(_input)
        print(greedy_str)
        print('----------')
        _refs = list()
        for ref in refs:
            _ref = decode(ref)
            _refs.append(_ref)

        greedy_test_decode_results['content'].append({'input': _input, 'output': greedy_str, 'refs': _refs})

    with open(os.path.join(log_dir, 'greedy_gen_examples_FIXED.json'), 'w') as f:
        json.dump(greedy_test_decode_results, f)
