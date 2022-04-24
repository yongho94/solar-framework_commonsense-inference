import sys, os

sys.path.append(os.getcwd())

import json
import yaml
from tqdm import tqdm
import argparse
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from distutils.util import strtobool as _bool
from torch.utils.data import DataLoader

from src.utils import load_logger, load_yaml

from src.finetune.finetune_utils import *
from src.finetune.finetune_trainer import FineTuneTrainer
from models import bart
from src.lr_schedule import WarmupLinearScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--file_dir', type=str, default='/mnt/data/user8/solar-commonsense_inference')
parser.add_argument('--dataset_type', type=str, default='conceptnet')
parser.add_argument('--model_name', type=str, default='bart', help="bart | gpt2")
parser.add_argument('--model_size', type=str, default='large', help="base | large")
parser.add_argument('--exp_type', type=str, default='baseline', help='baseline | experiments')
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--load_model', type=str, default='v07')
parser.add_argument('--use_greedy', type=_bool, default=False)
parser.add_argument('--use_beam', type=int, default=10 )

args = parser.parse_args()

np.random.seed(args.random_seed)

use_greedy = args.use_greedy
use_beam = args.use_beam > 1
beam_num = args.use_beam

log_dir = f'{args.file_dir}/log_fntn/{args.dataset_type}/{args.model_name}-{args.model_size}_{args.exp_type}/{args.load_model}'

main_config_path = f'{log_dir}/main_config.json'
tknz_config_path = f'{log_dir}/tknz_config.json'

dataset_config_path = f'config/{args.dataset_type}/datasets.yml'

print(f'Log Location : {log_dir}')
logging_dir = log_dir + '/inf_logging.log'
tb_dir = log_dir + '/tb'
gen_dir = log_dir + '/gen'
eval_dir = log_dir + '/eval'
ckpt_dir = log_dir + '/ckpt'

if not os.path.exists(log_dir):
    raise Exception

logger = load_logger(logging_dir, args.log_level)
logger.info('Logger is Successfully initialized !')

main_config = load_yaml(main_config_path)
tknz_config = load_yaml(tknz_config_path)
dataset_cfg = load_yaml(dataset_config_path)

model_cfg = main_config['model']
opt_cfg = main_config['opt']
log_cfg = main_config['log']

usable_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if usable_cuda else "cpu")


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
    raise NotImplementedError

else:
    raise NotImplementedError

_tokenizer = Tokenizer.from_pretrained(model_cfg['pretrained_model'])
tokenizer = CSKGTokenizer(_tokenizer, tknz_config)

vocab_len = len(tokenizer) + 1

with open(os.path.join(ckpt_dir, 'ckpt_loss.json'), 'r') as f:
    ckpt_loss = json.load(f)

best_ckpt = None
best_loss = 10000

for _ckpt, _loss in ckpt_loss.items():
    if best_loss > _loss:
        best_loss = _loss
        best_ckpt = _ckpt

logger.info("Model will be loaded from : {}".format(best_ckpt))

try:
    if str(device) == 'cpu':
        model = torch.load(best_ckpt, map_location=torch.device('cpu'))
    else:
        model = torch.load(best_ckpt)

except:
    best_ckpt = load_model.replace('log_fntn', 'log2_fntn')
    model = torch.load(best_ckpt)

gen_model = model2gen(model, model_cfg['pretrained_model']).to(device)

gen_model.to(device)
gen_model.eval()

decode = tokenizer.tokenizer.decode


dataset = load_fntn_datasets(dataset_cfg, tokenizer, logger)
eval_test_dataset = get_eval_dataset(dataset['test'], tokenizer, logger, 'test', 'dec')

test_decode_results = {
    'info': {'log': log_dir, 'ckpt': best_ckpt},
    'content': list()}

from copy import deepcopy

if use_greedy:
    greedy_results = deepcopy(test_decode_results)
    greedy_results['info']['decode_method'] = 'greedy'

if use_beam:
    beam_results = deepcopy(test_decode_results)
    beam_results['info']['decode_method'] = f'beam{beam_num}'


for sample in tqdm(eval_test_dataset, ncols=130):
    src = sample['src']
    refs = sample['ref']

    _input = decode(src)
    _refs = list()
    for ref in refs:
        _ref = decode(ref)
        _refs.append(_ref)

    enc_input_ids = torch.tensor(src).to(device).view(1, -1)
    enc_att_masks = torch.ones_like(enc_input_ids).to(device)

    inputs = {'input_ids': enc_input_ids, 'attention_mask': enc_att_masks}
    if use_greedy:
        greedy_output = gen_model.generate(**inputs, early_stopping=True,
                                           bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
        greedy_str = decode(greedy_output.tolist()[0])
        greedy_results['content'].append({'input': _input, 'output': greedy_str, 'refs': _refs})

    if use_beam:
        beam_output = gen_model.generate(**inputs, num_beams=beam_num, num_return_sequences=beam_num,  early_stopping=True,
                                         bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id) #max_length=60,
        beam_str = [decode(beam.tolist()).replace('<pad>','') for beam in beam_output]
        beam_results['content'].append({'input': _input, 'output': beam_str, 'refs': _refs})

if use_greedy:
    with open(os.path.join(log_dir, 'greedy_gen_examples.json'), 'w') as f:
        json.dump(greedy_results, f)

if use_beam:
    with open(os.path.join(log_dir, f'beam-{beam_num}_gen_examples.json'), 'w') as f:
        json.dump(beam_results, f)
# with open(os.path.join(log_dir, 'beam5_gen_examples.json'), 'w') as f:
#     json.dump(beam5_test_decode_results, f)
#
# for sample in tqdm(eval_test_dataset, ncols=130):
#     src = sample['src']
#     refs = sample['ref']
#     enc_input_ids = torch.tensor(src).to(device).view(1, -1)
#     enc_att_masks = torch.ones_like(enc_input_ids).to(device)
#
#     inputs = {'input_ids': enc_input_ids, 'attention_mask': enc_att_masks}
#
#     print('\n\nstart')
#     greedy_output = gen_model.generate(**inputs)#, early_stopping=True)
#     try:
#         beam5_output = gen_model.generate(**inputs, num_beams=5,  early_stopping=True) #max_length=60,
#     except:
#         print('beam err occur')
#         beam5_output = gen_model.generate(**inputs, num_beams=5, early_stopping=True, max_length=60)
#     print('end')
#     greedy_str = decode(greedy_output.tolist()[0])
#     beam5_str = decode(beam5_output.tolist()[0])
#
#     if '<gen>' in greedy_str:
#         greedy_str = greedy_str[greedy_str.find('<gen>') + 1 + len('<gen>'):].strip()
#     if '<gen>' in beam5_str:
#         beam5_str = beam5_str[beam5_str.find('<gen>')+1 + len('<gen>'):].strip()
#
#     _input = decode(src)
#     _refs = list()
#     for ref in refs:
#         _ref = decode(ref)
#         _refs.append(_ref)
#
#     greedy_test_decode_results['content'].append({'input': _input, 'output': greedy_str, 'refs': _refs})
#     beam5_test_decode_results['content'].append({'input': _input, 'output': beam5_str, 'refs': _refs})
#
#
# with open(os.path.join(log_dir, 'greedy_gen_examples.json'), 'w') as f:
#     json.dump(greedy_test_decode_results, f)
#
# with open(os.path.join(log_dir, 'beam5_gen_examples.json'), 'w') as f:
#     json.dump(beam5_test_decode_results, f)
#
# # ----- TEST ----- #
#
# for i in range(10):
#     output = gen_model(**inputs).logits
#     last_toks = torch.argmax(output[:, -1, :], -1)
#     old_inputs = {key: val.tolist() for key, val in inputs.items()}
#     old_inputs['input_ids'][0].append(int(last_toks))
#     old_inputs['attention_mask'][0].append(1)
#     inputs = {key : torch.tensor(val).to(device) for key, val in old_inputs.items()}
#
#
# # -- Non LM model -- #
#
# for i in range(10):
#     output = model.forward_conditional_gen(input_ids=inputs['input_ids'], att_masks=inputs['attention_mask'])
#     last_toks = torch.argmax(output[:, -1, :], -1)
#     old_inputs = {key: val.tolist() for key, val in inputs.items()}
#     old_inputs['input_ids'][0].append(int(last_toks))
#     old_inputs['attention_mask'][0].append(1)
#     inputs = {key : torch.tensor(val).to(device) for key, val in old_inputs.items()}
