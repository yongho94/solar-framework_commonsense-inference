import torch


class RecTaskTrainerForEncDec:
    def __init__(self, model, device, loss_func, tb_period, writer, _type='train'):
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.tb_period = tb_period
        self.writer = writer
        self._type = _type

    def train(self, task_data, global_step, save=False):
        enc_input_ids = task_data['enc_input_ids'].to(self.device)
        enc_att_mask = task_data['enc_att_mask'].to(self.device)
        dec_input_ids = task_data['dec_input_ids'].to(self.device)
        dec_att_mask = task_data['dec_att_mask'].to(self.device)
        dec_label_ids = task_data['dec_label_ids'].to(self.device)

        lm_logits = self.model.forward_conditional_gen(enc_input_ids, enc_att_mask, dec_input_ids, dec_att_mask)

        B, S, V = lm_logits.shape

        loss = self.loss_func(lm_logits.view(B * S, V), dec_label_ids.view(-1)).view(B, S)
        loss = torch.sum(torch.mul(loss, dec_att_mask), -1) / torch.sum(dec_att_mask, -1)
        total_loss = torch.mean(loss)

        if (global_step % self.tb_period == 0) and save is True:
            self.writer.add_scalar('{}_rec_task/total_loss'.format(self._type), total_loss.item(), global_step)
            self.writer.flush()

        return total_loss


class MaskGenTaskTrainerForDec:
    def __init__(self, model, device, loss_func, tb_period, writer, _type='train'):
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.tb_period = tb_period
        self.writer = writer
        self._type = _type

    def train(self, task_data, global_step, save=False):
        input_ids = task_data['input_ids'].to(self.device)
        att_mask = task_data['att_mask'].to(self.device)
        label_ids = task_data['label_ids'].to(self.device)

        lm_logits = self.model.forward_conditional_gen(input_ids, att_mask)

        B, S, V = lm_logits.shape

        loss = self.loss_func(lm_logits.view(B * S, V), label_ids.view(-1)).view(B, S)
        loss = torch.sum(torch.mul(loss, att_mask), -1) / torch.sum(att_mask, -1)
        total_loss = torch.mean(loss)

        if (global_step % self.tb_period == 0) and save is True:
            self.writer.add_scalar('{}_mask_gen_task/total_loss'.format(self._type), total_loss.item(), global_step)
            self.writer.flush()

        return total_loss



class ConTaskTrainerBase:
    def __init__(self, model, device, distance_model, loss_func, tb_period, writer, share_proj_layer, args, _type='train'):
        self.model = model
        self.device = device
        self.distance_model = distance_model
        self.loss_func = loss_func
        self.tb_period = tb_period
        self.writer = writer
        self.share_s, self.share_ro = share_proj_layer
        self._type = _type

class ConTaskTrainerForEncDec(ConTaskTrainerBase):
    def train(self, task_data, global_step, save=False):
        enc_input_ids_s = task_data['enc_input_ids_s'].to(self.device)
        enc_input_ids_ro = task_data['enc_input_ids_ro'].to(self.device)
        enc_att_mask_s = task_data['enc_att_mask_s'].to(self.device)
        enc_att_mask_ro = task_data['enc_att_mask_ro'].to(self.device)
        dec_input_ids_s = task_data['dec_input_ids_s'].to(self.device)
        dec_input_ids_ro = task_data['dec_input_ids_ro'].to(self.device)
        dec_att_mask_s = task_data['dec_att_mask_s'].to(self.device)
        dec_att_mask_ro = task_data['dec_att_mask_ro'].to(self.device)
        pos_sample = task_data['pos_samples'].to(self.device)

        s_repre = self.model.forward_latent_feature(
            enc_input_ids_s, enc_att_mask_s, dec_input_ids_s, dec_att_mask_s)#, None, self.share_s)
        ro_repre = self.model.forward_latent_feature(
            enc_input_ids_ro, enc_att_mask_ro, dec_input_ids_ro, dec_att_mask_ro)#, None, self.share_ro)

        dist_s_ro = self.distance_model(s_repre, ro_repre)
        dist_s_s = self.distance_model(s_repre, s_repre)
        dist_ro_ro = self.distance_model(ro_repre, ro_repre)

        sro_loss, sro_pos_dists, sro_neg_dists = self.loss_func.get_loss(dist_s_ro, pos_sample)
        ss_loss, ss_pos_dists, ss_neg_dists = self.loss_func.get_loss(dist_s_s, pos_sample)
        roro_loss, roro_pos_dists, roro_neg_dists = self.loss_func.get_loss(dist_ro_ro, pos_sample)

        loss = sro_loss + ss_loss + roro_loss
        pos_dists = sro_pos_dists + ss_pos_dists + roro_pos_dists
        neg_dists = sro_neg_dists + ss_neg_dists + roro_neg_dists

        if (global_step % self.tb_period == 0) and save is True:
            self.writer.add_scalar('{}_con_task/total_loss'.format(self._type), float(loss), global_step)
            self.writer.add_scalar('{}_con_task/s-ro_loss'.format(self._type), float(sro_loss), global_step)
            self.writer.add_scalar('{}_con_task/s-s_loss'.format(self._type), float(ss_loss), global_step)
            self.writer.add_scalar('{}_con_task/ro-ro_loss'.format(self._type), float(roro_loss), global_step)
            self.writer.add_scalar('{}_con_task/total_pos_dists'.format(self._type), float(pos_dists), global_step)
            self.writer.add_scalar('{}_con_task/total_neg_dists'.format(self._type), float(neg_dists), global_step)
            self.writer.add_scalar('{}_con_task/total_pos-neg'.format(self._type), float(pos_dists - neg_dists), global_step)
            self.writer.add_scalar('{}_con_task/s-ro_pos-neg'.format(self._type), float(sro_pos_dists - sro_neg_dists), global_step)
            self.writer.add_scalar('{}_con_task/s-s_pos-neg'.format(self._type), float(ss_pos_dists - ss_neg_dists), global_step)
            self.writer.add_scalar('{}_con_task/ro-ro_pos-neg'.format(self._type), float(roro_pos_dists - roro_neg_dists), global_step)
            self.writer.flush()

        return loss


class ConTaskTrainerForDec(ConTaskTrainerBase):
    def train(self, task_data, global_step, save=False):
        input_ids_s = task_data['input_ids_s'].to(self.device)
        input_ids_ro = task_data['input_ids_ro'].to(self.device)
        att_mask_s = task_data['att_mask_s'].to(self.device)
        att_mask_ro = task_data['att_mask_ro'].to(self.device)
        pos_sample = task_data['pos_samples'].to(self.device)

        s_repre = self.model.forward_latent_feature(
            input_ids_s, att_mask_s)
        ro_repre = self.model.forward_latent_feature(
            input_ids_ro, att_mask_ro)

        dist_s_ro = self.distance_model(s_repre, ro_repre)
        dist_s_s = self.distance_model(s_repre, s_repre)
        dist_ro_ro = self.distance_model(ro_repre, ro_repre)

        sro_loss, sro_pos_dists, sro_neg_dists = self.loss_func.get_loss(dist_s_ro, pos_sample)
        ss_loss, ss_pos_dists, ss_neg_dists = self.loss_func.get_loss(dist_s_s, pos_sample)
        roro_loss, roro_pos_dists, roro_neg_dists = self.loss_func.get_loss(dist_ro_ro, pos_sample)

        loss = sro_loss + ss_loss + roro_loss
        pos_dists = sro_pos_dists + ss_pos_dists + roro_pos_dists
        neg_dists = sro_neg_dists + ss_neg_dists + roro_neg_dists

        if (global_step % self.tb_period == 0) and save is True:
            self.writer.add_scalar('{}_con_task/total_loss'.format(self._type), float(loss), global_step)
            self.writer.add_scalar('{}_con_task/s-ro_loss'.format(self._type), float(sro_loss), global_step)
            self.writer.add_scalar('{}_con_task/s-s_loss'.format(self._type), float(ss_loss), global_step)
            self.writer.add_scalar('{}_con_task/ro-ro_loss'.format(self._type), float(roro_loss), global_step)
            self.writer.add_scalar('{}_con_task/total_pos_dists'.format(self._type), float(pos_dists), global_step)
            self.writer.add_scalar('{}_con_task/total_neg_dists'.format(self._type), float(neg_dists), global_step)
            self.writer.add_scalar('{}_con_task/total_pos-neg'.format(self._type), float(pos_dists - neg_dists), global_step)
            self.writer.add_scalar('{}_con_task/s-ro_pos-neg'.format(self._type), float(sro_pos_dists - sro_neg_dists), global_step)
            self.writer.add_scalar('{}_con_task/s-s_pos-neg'.format(self._type), float(ss_pos_dists - ss_neg_dists), global_step)
            self.writer.add_scalar('{}_con_task/ro-ro_pos-neg'.format(self._type), float(roro_pos_dists - roro_neg_dists), global_step)
            self.writer.flush()

        return loss

def mean_list(items):
    if type(items[0]) not in (list, tuple):
        return sum(items) / len(items)

    vals = 0
    nums = 0
    for val, num in items:
        vals += val
        nums += num

    return vals/nums