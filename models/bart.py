from transformers import BartModel
import torch
from torch import nn
from torch.nn import functional as F
from models.head_proj_layer import *
from transformers import BartTokenizer, BartForConditionalGeneration, BartModel
from models.model_utils import convert_model
from torch.nn import Linear


class CometBART(BartModel):
    def modify_lm_heads(self, vocab_len, lm_head_dropout_p):
        self.resize_token_embeddings(vocab_len)
        self.lm_head = nn.Linear(self.config.hidden_size, vocab_len)
        self.lm_head_dropout = nn.Dropout(p=lm_head_dropout_p)

    def add_proj_layer(self, proj_options, proj_head_dropout_p, shared_proj_layer=True):
        self.proj_head_dropout = nn.Dropout(p=proj_head_dropout_p)
        self.proj_options = proj_options
        self.seq_method = proj_options['pool_method']
        if shared_proj_layer is True:
            self.proj_head = self._get_proj_head()
        else:
            self.proj_head_s = self._get_proj_head()
            self.proj_head_ro = self._get_proj_head()

    def _get_proj_head(self):
        proj_layer_type = self.proj_options['proj_layer_type']
        hidden_size = self.config.hidden_size

        if proj_layer_type == 'non-linear':
            head_layer = NonLinearHeadProjLayer(input_hidden_size)
        elif proj_layer_type == 'multi-head':
            sub_opt = self.proj_options[proj_layer_type]
            head_num = sub_opt['head_num']
            head_dim = int(hidden_size / head_num) if sub_opt['head_dim'] == -1 else sub_opt['head_dim']
            head_layer = MultiHeadProjLayer(hidden_size, head_num, head_dim)
        else:
            raise NotImplementedError

        return head_layer

    def forward_conditional_gen(self, enc_input_ids, enc_att_mask, dec_input_ids, dec_att_mask):
        outputs = super().forward(input_ids=enc_input_ids, attention_mask=enc_att_mask,
                           decoder_input_ids=dec_input_ids, decoder_attention_mask=dec_att_mask)

        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.lm_head_dropout(last_hidden_state)
        lm_logits = self.lm_head(last_hidden_state)

        return lm_logits

    def forward_latent_feature(self, enc_input_ids, enc_att_mask, dec_input_ids, dec_att_mask, for_s=None):
        outputs = super().forward(input_ids=enc_input_ids, attention_mask=enc_att_mask,
                           decoder_input_ids=dec_input_ids, decoder_attention_mask=dec_att_mask)

        seq_feature = self.get_sequence_feature(outputs, enc_att_mask)
        seq_feature = self.proj_head_dropout(seq_feature)
        latent_vec = self._forward_projection(seq_feature, for_s)

        return latent_vec

    def get_sequence_feature(self, outputs, enc_att_mask):
        if self.seq_method == 'dec_bos':
            dec_last_hidden_state = outputs.last_hidden_state
            seq_feature = self._get_seq_feature_from_dec_bos(dec_last_hidden_state)
        elif self.seq_method == 'mean_pool_enc':
            enc_last_hidden_state = outputs.encoder_last_hidden_state
            seq_feature = self._get_seq_feature_from_mean_pool_enc(enc_last_hidden_state, enc_att_mask)
        elif self.seq_method == 'all_joint':
            enc_last_hidden_state = outputs.encoder_last_hidden_state
            dec_last_hidden_state = outputs.last_hidden_state
            seq_feature = self._get_seq_feature_from_mean_pool_enc_dec(
                enc_last_hidden_state, enc_att_mask, dec_last_hidden_state)
        else:
            raise NotImplementedError

        return seq_feature

    def _get_seq_feature_from_dec_bos(self, dec_last_hidden_state):
        seq_feature = dec_last_hidden_state[:, 0, :]
        return seq_feature

    def _get_seq_feature_from_mean_pool_enc(self, enc_last_hidden_state, att_mask):
        seq_feature = enc_last_hidden_state * att_mask.unsqueeze(-1)  # (B, S, H)
        seq_feature = torch.sum(seq_feature, dim=1)  # (B, H)
        seq_feature = seq_feature / torch.sum(att_mask, -1, keepdim=True)
        return seq_feature

    def _get_seq_feature_from_mean_pool_enc_dec(self, enc_last_hidden_state, enc_att_mask, dec_last_hidden_state):
        seq_feature = enc_last_hidden_state * enc_att_mask.unsqueeze(-1)  # (B, S, H)
        seq_feature_with_dec_bos = torch.cat((seq_feature, dec_last_hidden_state[:, :1, :]),dim=1)
        seq_feature = torch.sum(seq_feature_with_dec_bos, dim=1)  # (B, H)
        seq_feature = seq_feature / (torch.sum(enc_att_mask, -1, keepdim=True)+1)
        return seq_feature

    def _forward_projection(self, sequence_feature, for_s):
        if for_s is None:
            latent_vec = self.proj_head(sequence_feature)
        elif for_s is True:
            latent_vec = self.proj_head_s(sequence_feature)
        elif for_s is False:
            latent_vec = self.proj_head_ro(sequence_feature)
        else:
            raise NotImplementedError

        return latent_vec


def convert_BARTModel_to_BartForConditionalGeneration(bart_model, params):
    device = bart_model.device
    # model_name = f'facebook/bart-{size}'
    gen_model = BartForConditionalGeneration.from_pretrained(params)
    vocab_len, hidden = bart_model.lm_head.weight.shape
    use_bias = False if bart_model.lm_head.bias is None else True
    gen_model.resize_token_embeddings(vocab_len)
    gen_model.lm_head = Linear(hidden, vocab_len, bias=use_bias)

    gen_model = convert_model(bart_model, gen_model).to(device)

    return gen_model
