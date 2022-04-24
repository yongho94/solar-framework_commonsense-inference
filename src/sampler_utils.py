import pandas as pd
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
import torch
import os

def load_raw_tsv(raw_path, tokenizer, logger, truncate):
    data = pd.read_csv(raw_path, delimiter='\t', header=None)
    BLK_TOKEN = tokenizer.blk_token
    triples = list()
    subj_len = truncate['subj_len']
    obj_len = truncate['obj_len']

    cached_path = raw_path[:-4] + '_cached_{}_{}.torch-pkl'.format(subj_len, obj_len)
    if os.path.exists(cached_path):
        logger.info('Load from {}'.format(cached_path))
        return torch.load(cached_path)

    for row in tqdm(data.iloc, desc='loading_raw_file...', ncols=70):

        s, r, o = row

        if pd.isna(s) or pd.isna(r) or pd.isna(o):
            continue
        s = s.replace('___', BLK_TOKEN)
        r = '<{}>'.format(r)
        if o != o:  # pass nan
            continue
        o = o.replace('___', BLK_TOKEN)

        if len(tokenizer(s)) > subj_len or len(tokenizer(o)) > obj_len:
            continue
        triples.append([s, r, o])

    logger.info('Total loaded raw samples : {}'.format(len(triples)))

    torch.save(triples, cached_path)

    return triples


def load_atomic_sim_pkl(raw_path, tokenizer, logger):
    logger.info('Load Similar matrix from {}'.format(raw_path))
    with open(raw_path, 'rb') as f:
        data = pkl.load(f)
    blk_token = tokenizer.blk_token
    output = deepcopy(data)

    for s, i in data['s2i'].items():
        s = s.replace('___', blk_token)
        output['s2i'][s] = i
        output['i2s'][i] = s
    return output
