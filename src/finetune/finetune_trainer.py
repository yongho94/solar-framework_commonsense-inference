import torch


def get_finetune_trainer(model, device, loss_func, tb_period, writer, model_type):
    if model_type == 'enc-dec':
        return FineTuneTrainerForEncDec(model, device, loss_func, tb_period, writer)
    elif model_type == 'dec':
        return FineTuneTrainerForDec(model, device, loss_func, tb_period, writer)


class FineTuneTrainer:
    def __init__(self, model, device, loss_func, tb_period, writer):
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.tb_period = tb_period
        self.writer = writer

    def train(self, sample, global_step, _type='train', save_tb=True):
        raise NotImplementedError


class FineTuneTrainerForEncDec(FineTuneTrainer):
    def __init__(self, model, device, loss_func, tb_period, writer):
        super(FineTuneTrainerForEncDec, self).__init__(model, device, loss_func, tb_period, writer)

    def train(self, sample, global_step, _type='train', save_tb=True):
        enc_input_ids = sample['enc_input_ids'].to(self.device)
        enc_att_masks = sample['enc_att_masks'].to(self.device)
        dec_input_ids = sample['dec_input_ids'].to(self.device)
        dec_att_masks = sample['dec_att_masks'].to(self.device)
        dec_label_ids = sample['dec_label_ids'].to(self.device)

        lm_logits = self.model.forward_conditional_gen(enc_input_ids, enc_att_masks, dec_input_ids, dec_att_masks)

        B, S, V = lm_logits.shape
        loss = self.loss_func(lm_logits.view(B * S, V), dec_label_ids.view(-1)).view(B, S)
        loss = torch.sum(loss * dec_att_masks, dim=-1) / torch.sum(dec_att_masks, dim=-1)
        loss = torch.mean(loss)
        if global_step % self.tb_period == 0 and _type == 'train' and save_tb:
            self.writer.add_scalar('train/loss'.format(_type), loss.item(), global_step)
            self.writer.flush()

        return loss


class FineTuneTrainerForDec(FineTuneTrainer):
    def __init__(self, model, device, loss_func, tb_period, writer):
        super(FineTuneTrainerForDec, self).__init__(model, device, loss_func, tb_period, writer)

    def train(self, sample, global_step, _type='train', save_tb=True):
        input_ids = sample['input_ids'].to(self.device)
        att_masks = sample['att_masks'].to(self.device)
        label_ids = sample['label_ids'].to(self.device)

        lm_logits = self.model.forward_conditional_gen(input_ids, att_masks)

        B, S, V = lm_logits.shape
        loss = self.loss_func(lm_logits.view(B * S, V), label_ids.view(-1)).view(B, S)
        loss = torch.sum(loss * att_masks, dim=-1) / torch.sum(att_masks, dim=-1)
        loss = torch.mean(loss)
        if global_step % self.tb_period == 0 and _type == 'train' and save_tb:
            self.writer.add_scalar('train/loss'.format(_type), loss.item(), global_step)
            self.writer.flush()

        return loss

