def adapt_commonsense_tokenizer(tokenizer, config):
    tokenizer.add_tokens([config['additional_tokens'][key] for key in config['additional_tokens']])
    tokenizer.add_tokens([tokens for tokens in config['relation_tokens']])
    tokenizer.add_special_tokens(config['special_tokens'])
    return tokenizer

class BaseCSKGTokenizer:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

        self.tokenizer.add_tokens([self.config['additional_tokens'][key] for key in self.config['additional_tokens']])
        self.tokenizer.add_tokens([tokens for tokens in self.config['relation_tokens']])
        self.tokenizer.add_special_tokens(self.config['special_tokens'])

        # Additional Token Ids
        self.gen_token = self.config['additional_tokens']['gen_token']
        self.gen_token_id = self.tokenize(self.gen_token)[0]
        self.blk_token = self.config['additional_tokens']['blk_token']
        self.blk_token_id = self.tokenize(self.blk_token)[0]
        self.none_token = self.config['additional_tokens']['none_token']
        self.none_token_id = self.tokenize(self.none_token)[0]
        self.gen_task_token = self.config['additional_tokens']['gen_task_token']
        self.gen_task_token_id = self.tokenize(self.gen_task_token)[0]

        # Contrastive Task Tokens
        self.con_task_token = self.config['additional_tokens']['con_task_token']
        self.con_task_token_id = self.tokenize(self.con_task_token)[0]
        self.con_task_token_s = self.config['additional_tokens']['con_task_token_for_s']
        self.con_task_token_s_id = self.tokenize(self.con_task_token_s)[0]
        self.con_task_token_ro = self.config['additional_tokens']['con_task_token_for_ro']
        self.con_task_token_ro_id = self.tokenize(self.con_task_token_ro)[0]

        # Reconstruct Task Tokens
        self.rec_task_token = self.config['additional_tokens']['rec_task_token']
        self.rec_task_token_id = self.tokenize(self.rec_task_token)[0]

        # Mask Generate Task Tokens
        self.mask_gen_task_token = self.config['additional_tokens']['rec_task_token']
        self.mask_gen_task_token_id = self.tokenize(self.rec_task_token)[0]

        # Special Token Ids
        self.bos_token = self.tokenizer.bos_token
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_id = self.tokenizer.sep_token_id

    def tokenize(self, sequence):
        raise NotImplementedError

    def __call__(self, sequence):
        return self.tokenize(sequence)

    def __len__(self):
        return len(self.tokenizer)


class BartCSKGTokenizer(BaseCSKGTokenizer):
    def __init__(self, tokenizer, config):
        super(BartCSKGTokenizer, self).__init__(tokenizer, config)

    def tokenize(self, seq):
        assert type(seq) is str
        return self.tokenizer(seq)['input_ids'][1:-1]


class Gpt2CSKGTokenizer(BaseCSKGTokenizer):
    def __init__(self, tokenizer, config):
        super(Gpt2CSKGTokenizer, self).__init__(tokenizer, config)

    def tokenize(self, seq):
        assert type(seq) is str
        return self.tokenizer(seq)['input_ids']

