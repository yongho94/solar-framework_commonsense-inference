from torch.utils.data import Dataset, DataLoader
import torch


def get_tokenizer_prefix(tag, tokenizer):
    if tag == 'bos_token_id':
        return tokenizer.bos_token_id
    elif tag == 'con_task_token_id':
        return tokenizer.con_task_token_id
    elif tag == 'con_task_token_s_id':
        return tokenizer.con_task_token_s_id
    elif tag == 'con_task_token_ro_id':
        return tokenizer.con_task_token_ro_id
    elif tag == 'den_task_token_id':
        return tokenizer.den_task_token_id
    elif tag == 'den_task_token_for_dec_id':
        return tokenizer.den_task_token_id
    else:
        raise NotImplementedError


class BaseTaskAdaptor(Dataset):
    def __init__(self, options, sampler, tokenizer, enc_len, dec_len, batch_size, data_len):
        super(BaseTaskAdaptor, self).__init__()
        self.options = options
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.batch_size = batch_size
        self.data_len = data_len

        self.use_task_prefix = self.options['use_task_prefix']
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.sep = self.tokenizer.sep_token_id
        self.non = self.tokenizer.none_token_id
        self.pad = self.tokenizer.pad_token_id
        self.gen = self.tokenizer.gen_token_id

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        raise NotImplementedError


class ConTaskAdaptor(BaseTaskAdaptor):
    def __init__(self, options, sampler, tokenizer, enc_len, dec_len, batch_size, data_len=1000000, group_num=-1):
        super(ConTaskAdaptor, self).__init__(options, sampler, tokenizer, enc_len, dec_len, batch_size, data_len)
        self.task_id = self.tokenizer.con_task_token_id
        self.group_num = group_num
        self.group_size = int(self.batch_size / self.group_num)
        self.group_one_hot = list()
        self.seq_len = self.enc_len + self.dec_len
        for i in range(group_num):
            self.group_one_hot.extend([i] * self.group_size)
        assert len(self.group_one_hot) == self.batch_size
        self.group_one_hot = torch.tensor(self.group_one_hot)

        self._init_options()

    def _init_options(self):
        self.s_prefix = get_tokenizer_prefix(self.options['dec_input_ids_s'], self.tokenizer)
        self.ro_prefix = get_tokenizer_prefix(self.options['dec_input_ids_ro'], self.tokenizer)
        if self.options['format'] == 'enc-dec':
            self.formatting = self._enc_dec_formatting
        elif self.options['format'] == 'dec':
            self.formatting = self._dec_formatting
        else:
            raise NotImplementedError

    def _enc_dec_formatting(self, samples):
        outputs = {
            'enc_input_ids_s': list(),
            'enc_input_ids_ro': list(),
            'enc_att_mask_s': list(),
            'enc_att_mask_ro': list(),
            'dec_input_ids_s': list(),
            'dec_input_ids_ro': list(),
            'dec_att_mask_s': list(),
            'dec_att_mask_ro': list(),
            'pos_samples': list()
        }

        for sample in samples:
            s = sample['input'][0]
            r, o = sample['label']
            _group = sample['group']

            enc_input_ids_s = [self.task_id] if self.use_task_prefix else []
            enc_input_ids_ro = [self.task_id] if self.use_task_prefix else []
            enc_input_ids_s.extend([self.bos] + s + [self.sep, self.non, self.sep, self.non, self.eos])
            enc_input_ids_ro.extend([self.bos] + [self.non, self.sep] + r + [self.sep] + o + [self.eos])
            enc_input_ids_s.extend([self.pad] * (self.enc_len - len(enc_input_ids_s)))
            enc_input_ids_ro.extend([self.pad] * (self.enc_len - len(enc_input_ids_ro)))

            enc_input_ids_s = torch.tensor(enc_input_ids_s[:self.enc_len])
            enc_input_ids_ro = torch.tensor(enc_input_ids_ro[:self.enc_len])

            enc_att_mask_s = torch.ones_like(enc_input_ids_s) * (enc_input_ids_s != self.pad)
            enc_att_mask_s *= (enc_att_mask_s != self.non)
            enc_att_mask_ro = torch.ones_like(enc_input_ids_ro) * (enc_input_ids_ro != self.pad)
            enc_att_mask_ro *= (enc_att_mask_ro != self.non)

            dec_input_ids_s = [self.s_prefix]
            dec_input_ids_ro = [self.ro_prefix]
            dec_input_ids_s.extend([self.pad] * (self.dec_len - len(dec_input_ids_s)))
            dec_input_ids_ro.extend([self.pad] * (self.dec_len - len(dec_input_ids_ro)))

            dec_input_ids_s = torch.tensor(dec_input_ids_s[:self.dec_len])
            dec_input_ids_ro = torch.tensor(dec_input_ids_ro[:self.dec_len])

            dec_att_mask_s = torch.ones_like(dec_input_ids_s) * (dec_input_ids_s != self.pad)
            dec_att_mask_s *= (dec_att_mask_s != self.non)
            dec_att_mask_ro = torch.ones_like(dec_input_ids_ro) * (dec_input_ids_ro != self.pad)
            dec_att_mask_ro *= (dec_att_mask_ro != self.non)
            pos_samples = torch.ones_like(self.group_one_hot) * (self.group_one_hot == _group)

            outputs['enc_input_ids_s'].append(enc_input_ids_s)
            outputs['enc_input_ids_ro'].append(enc_input_ids_ro)
            outputs['enc_att_mask_s'].append(enc_att_mask_s)
            outputs['enc_att_mask_ro'].append(enc_att_mask_ro)
            outputs['dec_input_ids_s'].append(dec_input_ids_s)
            outputs['dec_input_ids_ro'].append(dec_input_ids_ro)
            outputs['dec_att_mask_s'].append(dec_att_mask_s)
            outputs['dec_att_mask_ro'].append(dec_att_mask_ro)
            outputs['pos_samples'].append(pos_samples)

        try:
            outputs = {key: torch.stack(outputs[key]) for key in outputs}
        except:
            for key in outputs:
                items = outputs[key]
                print('-------------')
                print(key)
                for item in items:
                    print(len(item))
            exit()

        return outputs

    def _dec_formatting(self, samples):
        outputs = {
            'input_ids_s': list(),
            'input_ids_ro': list(),
            'att_mask_s': list(),
            'att_mask_ro': list(),
            'pos_samples': list()
        }

        for sample in samples:
            s = sample['input'][0]
            r, o = sample['label']
            _group = sample['group']

            input_ids_s = [self.task_id] if self.use_task_prefix else []
            input_ids_ro = [self.task_id] if self.use_task_prefix else []
            input_ids_s.extend([self.s_prefix] + s + [self.sep, self.non, self.sep, self.non, self.eos])
            input_ids_ro.extend([self.ro_prefix] + [self.non, self.sep] + r + [self.sep] + o + [self.eos])
            input_ids_s.extend([self.pad] * (self.seq_len - len(input_ids_s)))
            input_ids_ro.extend([self.pad] * (self.seq_len - len(input_ids_ro)))

            input_ids_s = torch.tensor(input_ids_s)
            input_ids_ro = torch.tensor(input_ids_ro)

            att_mask_s = torch.ones_like(input_ids_s) * (input_ids_s != self.pad)
            att_mask_s *= (att_mask_s != self.non)
            att_mask_ro = torch.ones_like(input_ids_ro) * (input_ids_ro != self.pad)
            att_mask_ro *= (att_mask_ro != self.non)

            pos_samples = torch.ones_like(self.group_one_hot) * (self.group_one_hot == _group)

            outputs['input_ids_s'].append(input_ids_s)
            outputs['input_ids_ro'].append(input_ids_ro)
            outputs['att_mask_s'].append(att_mask_s)
            outputs['att_mask_ro'].append(att_mask_ro)
            outputs['pos_samples'].append(pos_samples)

        outputs = {key: torch.stack(outputs[key]) for key in outputs}

        return outputs

    def __getitem__(self, idx):
        if idx >= self.data_len:
            raise IndexError

        while True:
            try:
                samples = self.sampler.get_task_batch_sample()
                assert len(samples) == self.batch_size
                out_samples = self.formatting(samples)
                break
            except:
                print("Error Occured at Sampling Contrastive Task !")
                print("Try again !")


        return out_samples


class RecInfShuTaskAdaptor(BaseTaskAdaptor):
    def __init__(self, options, sampler, tokenizer, enc_len, dec_len, batch_size, data_len):
        super(RecInfShuTaskAdaptor, self).__init__(options, sampler, tokenizer, enc_len, dec_len, batch_size, data_len)
        self.task_id = self.tokenizer.rec_task_token_id
        #self.seq_len = self.enc_len + self.dec_len # Only for Decoder formatting

        if self.options['format'] == 'enc-dec':
            self.formatting = self._enc_dec_formatting
        elif self.options['format'] == 'dec':
            self.formatting = self._dec_formatting

    def _enc_dec_formatting(self, sample):
        s, r, o = sample['input']
        sl, rl, ol = sample['label']
        enc_input_ids = [self.task_id] if self.use_task_prefix else []
        enc_input_ids.extend([self.bos] + s + [self.sep] + r + [self.sep] + o + [self.eos])
        enc_input_ids.extend([self.pad] * (self.enc_len - len(enc_input_ids)))

        enc_input_ids = torch.tensor(enc_input_ids[:self.enc_len])
        enc_att_mask = torch.ones_like(enc_input_ids) * (enc_input_ids != self.pad)

        dec_origin = []
        dec_origin.extend([self.bos] + sl + [self.sep] + rl + [self.sep] + ol + [self.eos])
        dec_origin.extend([self.pad] * (self.dec_len - len(dec_origin)))
        dec_origin = torch.tensor(dec_origin[:self.dec_len])
        dec_origin_mask = torch.ones_like(dec_origin) * (dec_origin != self.pad)

        dec_input_ids = dec_origin[:-1].clone().detach()
        dec_att_mask = dec_origin_mask[:-1].clone().detach()

        dec_label_ids = dec_origin[1:].clone().detach()

        outputs = {
            'enc_input_ids': enc_input_ids,
            'enc_att_mask': enc_att_mask,
            'dec_input_ids': dec_input_ids,
            'dec_att_mask': dec_att_mask,
            'dec_label_ids': dec_label_ids
        }
        return outputs

    def _dec_formatting(self, sample):
        s, r, o = sample['input']
        sl, rl, ol = sample['label']
        input_ids = [self.task_id] if self.use_task_prefix else []
        input_ids.extend([self.bos] + s + [self.sep] + r + [self.sep] + o + [self.gen])
        input_ids.extend(sl + [self.sep] + rl + [self.sep] + ol + [self.eos])
        input_ids.extend([self.pad] * (self.seq_len - len(input_ids)))

        input_ids = input_ids[:self.seq_len]
        label_ids = input_ids[1:]
        input_ids = input_ids[:-1]

        label_ids = torch.tensor(label_ids)
        input_ids = torch.tensor(input_ids)

        att_mask = torch.ones_like(input_ids) * (input_ids != self.pad)

        outputs = {
            'input_ids': input_ids,
            'att_mask': att_mask,
            'label_ids': label_ids
        }
        return outputs

    def __getitem__(self, idx):
        if idx >= self.data_len:
            raise IndexError
        # sample = self.sampler.get_task_sample()
        # outputs = self.formatting(sample)

        while True:
            try:
                sample = self.sampler.get_task_sample()
                outputs = self.formatting(sample)
                break
            except:
                print("Error Occured at Sampling RecInfShuTaskAdaptor Task !")
                print("Try again !")



        return outputs


class MaskGenTaskAdaptor(BaseTaskAdaptor):
    def __init__(self, options, sampler, tokenizer, enc_len, dec_len, batch_size, data_len, use_batched=False):
        super(MaskGenTaskAdaptor, self).__init__(options, sampler, tokenizer, enc_len, dec_len, batch_size, data_len)
        self.task_id = self.tokenizer.mask_gen_task_token_id
        self.dec_id = self.tokenizer.bos_token_id
        self.seq_len = self.enc_len + self.dec_len
        self.use_batched = use_batched

        if self.options['format'] == 'enc-dec':
            raise NotImplementedError
        elif self.options['format'] == 'dec':
            self.formatting = self._dec_formatting

    def _dec_formatting(self, sample):
        s, r, o = sample['input']
        label = sample['label'][0]

        input_ids = [self.task_id] if self.use_task_prefix else []
        input_ids.extend([self.bos] + s + [self.sep] + r + [self.sep] + o + [self.gen] + label + [self.eos])
        input_ids.extend([self.pad] * (self.seq_len - len(input_ids)))
        input_ids = input_ids[:self.seq_len]
        label_ids = input_ids[1:]
        input_ids = input_ids[:-1]

        label_ids = torch.tensor(label_ids)
        input_ids = torch.tensor(input_ids)

        att_mask = torch.ones_like(input_ids) * (input_ids != self.pad)

        outputs = {
            'input_ids': input_ids,
            'att_mask': att_mask,
            'label_ids': label_ids
        }
        return outputs

    def __getitem__(self, idx):
        if idx >= self.data_len:
            raise IndexError
        #sample = self.sampler.get_task_sample()
        #outputs = self.formatting(sample)

        while True:
            try:
                sample = self.sampler.get_task_sample()
                outputs = self.formatting(sample)
                break
            except:
                print("Error Occured at Sampling MaskGenTaskAdaptor Task !")
                print("Try again !")


        return outputs

