import time
import pickle as pkl
from collections import defaultdict
import torch
import numpy as np

from src.sampler_utils import load_raw_tsv, load_atomic_sim_pkl


def load_datasets(dataset_cfg, tokenizer, logger):
    logger.info('Target Dataset : {}'.format(dataset_cfg['name']))
    if dataset_cfg['name'] in ['atomic-2020', 'atomic', 'atomic-2020-exclude-none', 'conceptnet']:
        sim_mat = {key: load_atomic_sim_pkl(
            dataset_cfg['sim'][key], tokenizer, logger) for key in dataset_cfg['sim']}
        dataset = {key: load_raw_tsv(
            dataset_cfg['dir'][key], tokenizer, logger, dataset_cfg['truncate'])
            for key in dataset_cfg['dir']}
        return dataset, sim_mat
    else:
        raise NotImplementedError


def get_data_sampler(task_cfg, dataset, sim_mat, tokenizer, logger, args):
    outputs = {'train': dict(), 'dev': dict()}
    batch_size = args.batch_size

    task = 'con_task'
    if task_cfg.get(task) is not None:
        logger.info('Add Contrastive Task with weight : {}'.format(task_cfg[task]['weight']))
        for key in outputs:
            logger.info("select Contrastive Version 1")
            outputs[key][task] = ConTaskDataSampler(
                task_cfg, dataset[key], tokenizer, logger, key, batch_size, sim_mat[key])

    task = 'rec_inf_shu_task'
    if task_cfg.get(task) is not None:
        logger.info('Add Reconstruction Task (Infill and Shuffle) with weight : {}'.format(task_cfg[task]['weight']))
        for key in outputs:
            outputs[key][task] = RecInfShuTaskDataSampler(
                task_cfg, dataset[key], tokenizer, logger, key, batch_size)

    task = 'mask_gen_task'
    if task_cfg.get(task) is not None:
        logger.info('Add Reconstruction Task (Infill and Shuffle) with weight : {}'.format(task_cfg[task]['weight']))
        for key in outputs:
            outputs[key][task] = MaskGenTaskDataSampler(
                task_cfg, dataset[key], tokenizer, logger, key, batch_size)

    return outputs


class BaseDataSampler:
    def __init__(self, config, dataset, tokenizer, logger, data_type='train', name='Default'):
        self.config = config
        self.data_type = data_type
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.idxes = [i for i in range(len(self.dataset))]
        self.cursor = 0
        self.logger = logger
        self.name = name
        self.logger.info('{}-[{}] : initialized !'.format(self.name, self.data_type))
        self._init_cursor()
        self._shuffle()

    def _shuffle(self):
        if self.config['init_with_shuffle'] is True and self.data_type == 'train':
            np.random.shuffle(self.idxes)
            self.logger.debug('{} : shuffled!'.format(self.name))
            self.logger.debug(str([i for i in self.idxes[:10]]))

    def _init_cursor(self, seed=None):
        self.logger.debug('{} : cursor initialized !, cursor set to zero'.format(self.name))
        self.cursor = 0
        if seed is not None:
            self.idxes = [i for i in range(len(self.dataset))]
            np.random.seed(seed)
            self._shuffle()

    def _get_tuple(self):
        if self.cursor >= len(self.idxes):
            self._shuffle()
            self._init_cursor()
        _tuple_idx = self.idxes[self.cursor]
        _tuple = self.dataset[_tuple_idx]
        self.cursor += 1
        return _tuple

    def _tknz_tuple(self, _tuple):
        s, r, o = [self.tokenizer(elem) for elem in _tuple]
        return s, r, o

    def get_sample(self, tokenize=False):
        _tuple = self._get_tuple()
        if tokenize:
            _tuple = self._tknz_tuple(_tuple)
        self.logger.debug('get "{}" sample'.format(str(_tuple)))
        return _tuple

    def get_task_sample(self):
        raise NotImplementedError

    def get_task_batch_sample(self):
        raise NotImplementedError

    def add_equipment(self, **argv):
        raise NotImplementedError


class ConTaskDataSampler(BaseDataSampler):
    def __init__(self, config, dataset, tokenizer, logger, data_type, batch_size, subj_dist):
        name = "Contrastive Task Data Sampler"
        super(ConTaskDataSampler, self).__init__(config, dataset, tokenizer, logger, data_type, name)
        self.task_cfg = config['con_task']
        self.s2ro = self._generate_s2ro(dataset)

        self.batch_size = batch_size
        self.s2i = subj_dist['s2i']
        self.i2s = subj_dist['i2s']
        self.s2s = subj_dist['s2s']
        self.s2s_top3000 = subj_dist['s2s_top3000']

        # Setting Contrastive Learning Methodology
        self.logger.info('Use {} contrastive method'.format(self.task_cfg['method']))
        if self.task_cfg['method'] == 'naive':
            self.get_contrast_sample = self._get_naive_contrast_task_sample
        elif self.task_cfg['method'] == 'cluster':
            self.group_num = self.task_cfg['cluster_contrast']['group_num']
            self.group_size = int(self.batch_size / self.group_num)
            self.pos_subj_min = self.task_cfg['cluster_contrast']['pos_subj_min']
            assert self.group_num * self.group_size == self.batch_size
            self.get_contrast_sample = self._get_clustered_contrast_task_sample
            self.logger.info('Contrastive Group Num : {}'.format(self.group_num))
            self.logger.info('Contrastive Group Size : {}'.format(self.group_size))
        else:
            raise NotImplementedError

        # Setting Sampling Method
        self.logger.info('Use {} Sampling Method'.format(self.task_cfg['sampling_method']))
        if self.task_cfg['sampling_method'] == 'random':
            self.sampling_subject = self._subject_random_sampling
        elif self.task_cfg['sampling_method'] == 'adv':
            self.min_subj_sim = self.task_cfg['adv_sampling']['min_sim']
            self.max_subj_sim = self.task_cfg['adv_sampling']['max_sim']
            self.logger.info('{} < adv_subject < {}'.format(self.min_subj_sim, self.max_subj_sim))
            assert self.max_subj_sim - self.min_subj_sim > 0
            self.sampling_subject = self._subject_adv_sampling
        else:
            raise NotImplementedError

    def _generate_s2ro(self, dataset):
        s2ro = defaultdict(list)
        for s, r, o in dataset:
            s2ro[s].append((r, o))
        return s2ro

    def _get_naive_contrast_task_sample(self, root_subj_idx):
        subj_idx_list = self.sampling_subject(root_subj_idx, self.batch_size)
        samples = list()
        for subj_idx, group_num in subj_idx_list.items():
            sample = self._retrieve_own(subj_idx)
            samples.append({'sample': sample, 'group': group_num})
        return samples

    def _get_clustered_contrast_task_sample(self, root_subj_idx):
        anchor_subj_idx_list = self.sampling_subject(root_subj_idx, self.group_num)
        subj_idx_list = list()
        samples = list()
        for anchor_subj_idx, group_num in anchor_subj_idx_list.items():
            subj_idx_list.append((anchor_subj_idx, group_num))
            pos_subj_idx_list = self.sampling_pos_subjects(anchor_subj_idx)
            for pos_subj in pos_subj_idx_list:
                subj_idx_list.append((pos_subj, group_num))
        for subj_idx, group_num in subj_idx_list:
            sample = self._retrieve_own(subj_idx)
            samples.append({'sample': sample, 'group': group_num})
        return samples

    def _retrieve_own(self, subj_idx):
        s = self.i2s[subj_idx]
        ro_list = self.s2ro[s]
        ro_idx = np.random.choice(list(range(len(ro_list))))
        r, o = ro_list[ro_idx]
        return [s, r, o]

    def sampling_pos_subjects(self, anchor_subj_idx):
        targets = self.s2s_top3000['idx'][anchor_subj_idx][self.pos_subj_min < self.s2s_top3000['val'][anchor_subj_idx]]
        pos_subjects = np.random.choice(targets, size=self.group_size - 1, replace=True)
        return list(pos_subjects)

    def _subject_random_sampling(self, root_subj_idx, sample_num):
        subj_list = {root_subj_idx: 0}
        while len(subj_list) != sample_num:
            s, _, _ = self.get_sample()
            s_idx = self.s2i[s]
            if subj_list.get(s_idx) is None:
                subj_list[s_idx] = len(subj_list)
        return subj_list

    def _subject_adv_sampling(self, root_subj_idx, sample_num):
        self.logger.debug('Sampling "Adversarial" subjects, root subject is {}\n sampling numbers : {}'.format(root_subj_idx, sample_num))
        subj_list = {root_subj_idx: 0}
        ref_idx = root_subj_idx
        again = False
        limit_iter = 50
        limit_over_num = 0
        _iter = 0
        _occur_time = 0
        _occur_len = 0
        while len(subj_list) != sample_num:
            self.logger.debug("Reference Sample Idx is : {}".format(ref_idx))
            targets = self.s2s_top3000['idx'][ref_idx][
                (self.min_subj_sim < self.s2s_top3000['val'][ref_idx]) &
                (self.s2s_top3000['val'][ref_idx] < self.max_subj_sim)]
            if len(targets) == 0 or _iter > limit_iter:
                _occur_len = len(targets)
                limit_over_num += 1
                ref_idx = np.random.choice(len(self.s2s_top3000['idx']))
                _iter = 0
                if _occur_time == 0:
                    _occur_time = len(subj_list)
                continue
            candidate_idx = np.random.choice(targets)
            self.logger.debug("Adversarial Sample pick : {}".format(candidate_idx))
            for key in subj_list:
                if self.s2s[candidate_idx][key] > self.max_subj_sim or candidate_idx in subj_list:
                    self.logger.debug("is rejected, because exceed {}, picked : {} or already picked".format(
                        self.max_subj_sim, self.s2s[candidate_idx][key]
                    ))
                    _iter += 1
                    again = True
                    break
                    
            if again:
                again = False
                continue

            subj_list[candidate_idx] = len(subj_list)
            ref_idx = candidate_idx
        self.logger.debug("LIMIT OVER : {}/{} at first {} - {}".format(limit_over_num, sample_num, _occur_time, _occur_len))
        self.logger.debug("Adversarial sample passed! \nNow:{}".format(subj_list))
        return subj_list

    def get_task_sample(self):
        raise NotImplementedError

    def get_task_batch_sample(self):
        start_time = time.time()
        root_subj_idx = None
        while root_subj_idx is None:
            s, _, _ = self.get_sample(tokenize=False)
            root_subj = s
            root_subj_idx = self.s2i[root_subj]
            if len(self.s2ro[root_subj]) == 0:
                root_subj_idx = None

        contrast_samples = self.get_contrast_sample(root_subj_idx)
        samples = list()
        for sample in contrast_samples:
            s, r, o = [self.tokenizer(i) for i in sample['sample']]
            group = sample['group']
            _input = [s]
            _label = [r, o]
            samples.append({'input': _input,
                            'label': _label,
                            'group': group})
        req_time = time.time() - start_time
        self.logger.debug("Required time to generate Contrastive Task samples : {}".format(req_time))
        return samples


class RecInfShuTaskDataSampler(BaseDataSampler):
    def __init__(self, config, dataset, tokenizer, logger, data_type, batch_size):
        name = "Reconstruct from Infilled & Shuffled Task Data Sampler"
        super(RecInfShuTaskDataSampler, self).__init__(config, dataset, tokenizer, logger, data_type, name)
        self.task_cfg = self.config['rec_inf_shu_task']
        if self.task_cfg['method'] == 'naive':
            self.logger.info('Reconstruct from Infilled & Shuffled Task Data Method : {}'.format(self.task_cfg['method']))
            self.get_task_sample = self._get_naive_task_sample
        else:
            raise NotImplementedError
        self.crpt_probs = {'token_crpt':
                               [self.task_cfg['no_crpt_prob'], self.task_cfg['subj_crpt_prob'],
                                self.task_cfg['rel_crpt_prob'], self.task_cfg['obj_crpt_prob']],
                           'sen_crpt':self.task_cfg['rel_crpt_prob']}

        self.logger.info('Probabilities of corruption\n\t '
                         '* No corrupt Probability : {}\n\t '
                         '* Subject corrupt Probability : {}\n\t '
                         '* Relation corrupt Probability : {}\n\t '
                         '* Object corrupt Probability : {}\n\t '
                         '* Sentence Shuffling Probability : {}'.format(
                        self.crpt_probs['token_crpt'][0], self.crpt_probs['token_crpt'][1], self.crpt_probs['token_crpt'][2],
                        self.crpt_probs['token_crpt'][3], self.crpt_probs['sen_crpt']))

        self.batch_size = batch_size

    def _get_naive_task_sample(self):
        s, r, o = self.get_sample(tokenize=True)
        _label = s.copy(), r.copy(), o.copy()
        token_crpt = np.random.choice([0, 1, 2, 3])
        sen_crpt = np.random.uniform() > self.crpt_probs['sen_crpt']

        if token_crpt == 1:  # Subject mask
            span_len = np.random.poisson(3)
            start_point = np.random.randint(len(s))
            end_point = start_point + span_len

            new_s = []
            for i in range(len(s)):
                if i == start_point:
                    new_s.append(self.tokenizer.mask_token_id)
                if start_point <= i < end_point:
                    continue
                new_s.append(s[i])
            s = new_s

        elif token_crpt == 2:  # Relation Mask
            r[0] = self.tokenizer.mask_token_id

        elif token_crpt == 3:  # Object Mask
            span_len = np.random.poisson(3)
            start_point = np.random.randint(len(o))
            end_point = start_point + span_len

            new_o = []
            for i in range(len(o)):
                if i == start_point:
                    new_o.append(self.tokenizer.mask_token_id)
                if start_point <= i < end_point:
                    continue
                new_o.append(o[i])
            o = new_o

        _input = [s, r, o]
        if sen_crpt:
            np.random.shuffle(_input)

        task_sample = {'input': _input,
                       'label': _label}

        return task_sample

    def get_task_sample(self):
        return self.get_task_sample()

    def get_task_batch_sample(self):
        samples = list()
        for _ in range(self.batch_size):
            samples.append(self.get_task_sample())
        return samples


class MaskGenTaskDataSampler(BaseDataSampler):
    def __init__(self, config, dataset, tokenizer, logger, data_type, batch_size):
        name = "Generate Masked Token task Sampler"
        super(MaskGenTaskDataSampler, self).__init__(config, dataset, tokenizer, logger, data_type, name)
        self.task_cfg = self.config['mask_gen_task']
        if self.task_cfg['method'] == 'naive':
            self.logger.info('Generate Masked Token task  : {}'.format(self.task_cfg['method']))
            self.get_task_sample = self._get_naive_task_sample
        else:
            raise NotImplementedError

        self.mask_probs = {'mask_probs':
                   [self.task_cfg['subj_mask_prob'], self.task_cfg['rel_mask_prob'], self.task_cfg['obj_mask_prob']]}

        self.logger.info('Probabilities of masking\n\t '
                         '* Subject mask Probability : {}\n\t '
                         '* Relation mask Probability : {}\n\t '
                         '* Object mask Probability : {}\n\t '.format(
                        self.mask_probs['mask_probs'][0],
                        self.mask_probs['mask_probs'][1],
                        self.mask_probs['mask_probs'][2]))

        self.batch_size = batch_size

    def _get_naive_task_sample(self):
        s, r, o = self.get_sample(tokenize=True)
        _backup_sro = [s.copy(), r.copy(), o.copy()]
        token_mask = np.random.choice([0, 1, 2], p=self.mask_probs['mask_probs'])

        if token_mask == 0:  # Subject mask
            s = [self.tokenizer.mask_token_id]
            _label = [_backup_sro[0]]

        elif token_mask == 1:  # Relation Mask
            r = [self.tokenizer.mask_token_id]
            _label = [_backup_sro[1]]

        elif token_mask == 2:  # Object Mask
            o = [self.tokenizer.mask_token_id]
            _label = [_backup_sro[2]]
        else:
            raise Exception

        _input = [s, r, o]

        task_sample = {'input': _input,
                       'label': _label}

        return task_sample

    def get_task_sample(self):
        return self.get_task_sample()

    def get_task_batch_sample(self):
        samples = list()
        for _ in range(self.batch_size):
            samples.append(self.get_task_sample())
        return samples
