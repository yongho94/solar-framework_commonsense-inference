from src.sampler_utils import load_raw_tsv
from torch.utils.data import Dataset
from src.sampler import BaseDataSampler
import random
import torch

random.seed(42)


def load_fntn_datasets(dataset_cfg, tokenizer, logger):
    logger.info('Target Dataset : {}'.format(dataset_cfg['name']))
    dataset = {key: load_raw_tsv(dataset_cfg['dir'][key], tokenizer, logger, dataset_cfg['truncate'])
               for key in dataset_cfg['dir']}

    return dataset


def get_finetune_dataset(dataset, dataset_cfg, tokenizer, logger, _type, model_type='enc-dec'):
    if model_type == 'enc-dec':
        fntn_dataset = FineTuneDatasetForEncDec(dataset, tokenizer, dataset_cfg['truncate'])
    elif model_type == 'dec':
        fntn_dataset = FineTuneDatasetForDec(dataset, tokenizer, dataset_cfg['truncate'])
    else:
        raise NotImplementedError
    logger.info("Load Fine-tuning datasets [{}] : {}".format(_type, len(fntn_dataset)))
    return fntn_dataset


def get_eval_dataset(dataset, tokenizer, logger, _type, model_type='enc-dec'):
    triple_dict = dict()
    for row in dataset:
        s, r, o = row
        key = s + '||' + r
        if triple_dict.get(key) is None:
            triple_dict[key] = list()
        triple_dict[key].append(o)

    outputs = list()
    for key in triple_dict:
        src = key.split('||')
        ref = list(set(triple_dict[key]))
        outputs.append({'src': src, 'ref': ref})

    logger.info("Load Evaluation datasets [{}] : {}".format(_type, len(outputs)))

    if model_type == 'enc-dec':
        return EvalDatasetForEncDec(outputs, tokenizer, logger)
    elif model_type == 'dec':
        return EvalDatasetForDec(outputs, tokenizer, logger)
    else:
        raise NotImplementedError


class EvalDataset(Dataset):
    def __init__(self, dataset, tokenizer, logger):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.logger = logger
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.sep = self.tokenizer.sep_token_id
        self.gen = self.tokenizer.gen_token_id

    def formatting(self, sample):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.formatting(sample)


class EvalDatasetForEncDec(EvalDataset):
    def __init__(self, dataset, tokenizer, logger):
        super(EvalDatasetForEncDec, self).__init__(dataset, tokenizer, logger)

    def formatting(self, sample):
        dp = {'src': None, 'ref': list()}

        src, ref_list = sample['src'], sample['ref']
        s, r = [self.tokenizer(i) for i in src]
        _input = [self.bos] + s + [self.sep] + r + [self.sep] + [self.gen] + [self.eos]
        dp['src'] = _input
        for ref in ref_list:
            ref_token = self.tokenizer(ref)
            _output = [self.bos] + ref_token + [self.eos]
            dp['ref'].append(_output)

        if len(dp['ref']) == 0:
            raise Exception

        return dp


class EvalDatasetForDec(EvalDataset):
    def __init__(self, dataset, tokenizer, logger):
        super(EvalDatasetForDec, self).__init__(dataset, tokenizer, logger)

    def __len__(self):
        return len(self.dataset)

    def formatting(self, sample):
        dp = {'src': None, 'ref': list()}

        src, ref_list = sample['src'], sample['ref']
        s, r = [self.tokenizer(i) for i in src]
        _input = [self.bos] + s + [self.sep] + r + [self.sep] + [self.gen]
        dp['src'] = _input
        for ref in ref_list:
            ref_token = self.tokenizer(ref)
            _output = ref_token + [self.eos]
            dp['ref'].append(_output)

        if len(dp['ref']) == 0:
            raise Exception

        return dp


class FineTuneDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        super(FineTuneDataset, self).__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.gen = self.tokenizer.gen_token_id
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.sep = self.tokenizer.sep_token_id
        self.pad = self.tokenizer.pad_token_id
        self.shuffle()

    def __len__(self):
        return len(self.dataset)

    def formatting(self, sample):
        raise NotImplementedError

    def shuffle(self):
        random.shuffle(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = self.formatting(sample)
        return sample


class FineTuneDatasetForEncDec(FineTuneDataset):
    def __init__(self, dataset, tokenizer, truncate):
        super(FineTuneDatasetForEncDec, self).__init__(dataset, tokenizer)
        self.enc_len, self.dec_len = truncate['subj_len'] + 6, truncate['obj_len'] + 5

    def formatting(self, sample):
        s, r, o = sample
        s_tokens = self.tokenizer(s)
        r_tokens = self.tokenizer(r)
        o_tokens = self.tokenizer(o)

        _input = [self.bos]
        _input.extend(s_tokens + [self.sep])
        _input.extend(r_tokens + [self.sep])
        _input.extend([self.gen])
        _input.extend([self.pad] * (self.enc_len - len(_input)))

        _output = [self.bos] + o_tokens + [self.eos]
        _output.extend([self.pad] * (self.dec_len - len(_output)))

        _enc_input_ids = torch.tensor(_input)
        _enc_att_masks = torch.ones_like(_enc_input_ids) * (_enc_input_ids != self.pad)
        _dec_origin = torch.tensor(_output)
        _dec_input_ids = _dec_origin[:-1].clone().detach()
        _dec_att_masks = torch.ones_like(_dec_input_ids) * (_dec_input_ids != self.pad)
        _dec_output_ids = _dec_origin[1:].clone().detach()

        output = {'enc_input_ids': _enc_input_ids,
                  'enc_att_masks': _enc_att_masks,
                  'dec_input_ids': _dec_input_ids,
                  'dec_att_masks': _dec_att_masks,
                  'dec_label_ids': _dec_output_ids}

        return output


class FineTuneDatasetForDec(FineTuneDataset):
    def __init__(self, dataset, tokenizer, truncate):
        super(FineTuneDatasetForDec, self).__init__(dataset, tokenizer)
        self.dec_len = truncate['subj_len'] + 6 + truncate['obj_len'] + 5

    def __len__(self):
        return len(self.dataset)

    def formatting(self, sample):
        s, r, o = sample
        s_tokens = self.tokenizer(s)
        r_tokens = self.tokenizer(r)
        o_tokens = self.tokenizer(o)

        _input = [self.bos]
        _input.extend(s_tokens + [self.sep])
        _input.extend(r_tokens + [self.sep, self.gen])
        _input.extend(o_tokens + [self.eos])

        _input.extend([self.pad] * (self.dec_len - len(_input)))
        _input_ids = _input[:-1]
        _label_ids = _input[1:]

        _input_ids = torch.tensor(_input_ids)
        _label_ids = torch.tensor(_label_ids)

        _att_masks = torch.ones_like(_input_ids) * (_input_ids != self.pad)
        output = {'input_ids': _input_ids,
                  'att_masks': _att_masks,
                  'label_ids': _label_ids}

        return output

