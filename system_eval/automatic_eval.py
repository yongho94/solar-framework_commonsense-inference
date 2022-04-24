import argparse
import os
import sys
import json
sys.path.append(os.path.join(os.getcwd(), 'system_eval'))

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import read_jsonl, remove_prefix
from evaluation.eval import QGEvalCap
from tabulate import tabulate
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_refs_preds(l, type=1):
    if type==1:
        tails = l["fact"]["tails"]
        head = l["fact"]["head"]
        prompt = l["prompt"]
        generations = l["generations"]
        gens = [remove_prefix(g, prompt).strip() for g in generations]
    if type==2:
        tails = l["refs"]
        head = l["input"]
        gens = [l["output"]]
    if type==3:
        tails = l["fact"]["tails"]
        head = l["fact"]["head"]
        gens = l["generations"]
    return gens, tails, head

def get2(l):
    return list(zip(*l))[1]


def topk_eval(model_name, data, data_type, k):
    topk_gts = {}
    topk_res = {}
    topk_exact_match = []
    topk_exact_match_not_none = []
    topk_bleu_score = []

    topk_is_head = []
    func = SmoothingFunction()
    # for i, l in enumerate(data):
    for i, l in tqdm(enumerate(data)):
        (gens, tails, head) = get_refs_preds(l, type=data_type)
        gens = [g.replace('<s>', '').replace('</s>', '').strip() for g in gens]
        tails = [t.replace('<s>', '').replace('</s>', '').strip() for t in tails]
        head = head.replace('<s>', '').replace('</s>', '').strip()
        sentence_tails = [t.lower() for t in tails]
        split_tails = [t.lower().replace('<s>', '').replace('</s>', '').split() for t in tails]

        for (j, g) in enumerate(gens[:k]):
            key = str(i) + "_" + str(j)
            topk_gts[key] = sentence_tails
            topk_res[key] = [g.lower()]

            b = sentence_bleu(split_tails, g.lower().split(), weights=(0.5, 0.5), smoothing_function=func.method1)
            topk_bleu_score.append((l, b))
            if g in sentence_tails:
                topk_exact_match.append((l, 1))
                if g != "none":
                    topk_exact_match_not_none.append((l, 1))
            else:
                topk_exact_match.append((l, 0))
                if g != "none":
                    topk_exact_match_not_none.append((l, 0))
            if g == head:
                topk_is_head.append((l, 1))
            else:
                topk_is_head.append((l, 0))

    print("---------------TOP K={}---------------".format(k))
    #print(np.mean(get2(topk_exact_match)))
    #print(np.mean(get2(topk_exact_match_not_none)))
    print(np.mean(get2(topk_bleu_score)))
    QGEval = QGEvalCap(model_name, topk_gts, topk_res)
    score, scores = QGEval.evaluate()
    scores["Exact_match"] = np.mean(get2(topk_exact_match))
    #scores["TailIsHead"] = np.mean(get2(topk_is_head))
    return score, topk_bleu_score, scores


def eval(data_file, data_type, model_name):
    data = read_jsonl(data_file)
    return topk_eval(model_name, data, data_type, k=1)

def toRow(name, results, columns):
    return [name] + [format(float(results[c]), '#.3f') for c in columns]

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='pycharm')
parser.add_argument('--file_dir', type=str, default='/mnt/data/user8/solar-commonsense_inference/log_fntn')
parser.add_argument('--dataset_type', type=str, default='atomic')
parser.add_argument('--model_name', type=str, default='bart')
parser.add_argument('--model_size', type=str, default='large')
parser.add_argument('--exp_type', type=str, default='baseline')
parser.add_argument('--target', type=str, default=None)

args = parser.parse_args()

targets_list = list()

target_path = f'{args.file_dir}/{args.dataset_type}/{args.model_name}-{args.model_size}_{args.exp_type}'
if args.target is None:
    target_list = os.listdir(target_path)
else:
    target_list = [args.target]

target_list.sort()

decode_type = ['greedy']
print(target_list)

#target_list = target_list[:7]
for target_file in target_list:
    for decode in decode_type:
        input_file = f'{target_path}/{target_file}/{decode}_gen_examples.json'
        output_file = f'{target_path}/{target_file}/eval/{decode}_results.txt'
        results_per_sample_file = f'{target_path}/{target_file}/eval/{decode}_results_per_sample.pkl'
        try:
            with open(input_file, 'r') as f:
                input_file = json.load(f)
        except:
            continue
        # Eval
        print('TEST target : {}'.format(input_file['info']['ckpt']))
        print(f'Decoded Type {decode}')
        gen_data = input_file['content']

        scores, topk_bleu_score, score_list = topk_eval(model_name='BART-ATOMIC2020', data=gen_data, data_type=2, k=1)

        results_per_sample = list()

        for idx, sample in enumerate(gen_data):
            sample_result = dict()
            sample_result.update(sample)

            for key in score_list:
                if type(score_list[key]) is not list:
                    continue
                val = score_list[key][idx]
                sample_result[key] = val

            results_per_sample.append(sample_result)

        print(scores)
        for key in scores:
            print(round(float(scores[key]) * 100, 4), end='\t')

        with open(output_file, 'w') as f:
            for key, item in scores.items():
                f.write('{} : {}\n'.format(key, item))

        print('\n\n')

        import pickle as pkl
        with open(results_per_sample_file, 'wb') as f:
            pkl.dump(results_per_sample, f)