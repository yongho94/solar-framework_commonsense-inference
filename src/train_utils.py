from src.feed_model import *
from torch.utils.data import DataLoader


def get_data_feeder_from_sampler(sampler, options, tokenizer, task_cfg, batch_size, iter_per_epoch):
    data_feeder = dict()
    enc_len = task_cfg['max_enc'] + 5
    dec_len = task_cfg['max_dec'] + 5
    for task_key, task_sampler in sampler.items():
        for option_key, val in options['common'].items():
            options[task_key][option_key] = val


        if task_key == 'con_task':
            if task_cfg['con_task']['method'] == 'cluster':
                group_num = task_cfg['con_task']['cluster_contrast']['group_num']
            elif task_cfg['con_task']['method'] == 'naive':
                group_num = batch_size
            else:
                raise NotImplementedError

            task_adaptor = ConTaskAdaptor(
                options[task_key], task_sampler, tokenizer, enc_len, dec_len, batch_size, iter_per_epoch, group_num)
            data_feeder[task_key] = task_adaptor

        elif task_key == 'rec_inf_shu_task':
            task_adaptor = RecInfShuTaskAdaptor(
                options[task_key], task_sampler, tokenizer, enc_len, dec_len, batch_size, iter_per_epoch * batch_size)
            data_feeder[task_key] = task_adaptor

        elif task_key == 'mask_gen_task':
            task_adaptor = MaskGenTaskAdaptor(
                options[task_key], task_sampler, tokenizer, enc_len, dec_len, batch_size, iter_per_epoch * batch_size)
            data_feeder[task_key] = task_adaptor

    return data_feeder


class WrappedLoader:
    def __init__(self, dataset, batch_size):
        self.loader = DataLoader(dataset,batch_size=batch_size, drop_last=False, shuffle=True, num_workers=10)
        pass

    def __getitem__(self, idx):
        for data in self.loader:
            yield data
