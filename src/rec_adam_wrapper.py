from src.rec_adam import RecAdam, anneal_function
from transformers import get_linear_schedule_with_warmup

##########
'''
#params = [p for n, p in model.named_parameters()]
pretrained_weights = anneal_targets
new_model = model
no_decay = ["bias", "layer_norm.weight"]
for n, p in model.named_parameters():
    for nd in no_decay:
        if nd in n:
            print(n)

for n, p in new_model.named_parameters():
    if not any(nd in n for nd in no_decay):
        print(n)
        if n in pretrained_weights:
            print(n)
pretrained_weights.keys()

any(nd in n for nd in no_decay)
[p for n, p in new_model.named_parameters() if
                       not any(nd in n for nd in no_decay) and n in pretrained_weights]
'''
##########

def get_adam_optimizer(new_model, pretrained_model, pretrained_weights, args, t_total):
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in new_model.named_parameters() if
                       not any(nd in n for nd in no_decay) and n in pretrained_weights],
            "weight_decay": args.weight_decay,
            "anneal_w": args.recadam_anneal_w,
            "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                not any(nd in p_n for nd in no_decay) and p_n in pretrained_weights]
        },
        {
            "params": [p for n, p in new_model.named_parameters() if
                       not any(nd in n for nd in no_decay) and n not in pretrained_weights],
            "weight_decay": args.weight_decay,
            "anneal_w": 0.0,
            "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                not any(nd in p_n for nd in no_decay) and p_n not in pretrained_weights]
        },
        {
            "params": [p for n, p in new_model.named_parameters() if
                       any(nd in n for nd in no_decay) and n in pretrained_weights],
            "weight_decay": 0.0,
            "anneal_w": args.recadam_anneal_w,
            "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                any(nd in p_n for nd in no_decay) and p_n in pretrained_weights == p_n]
        },
        {
            "params": [p for n, p in new_model.named_parameters() if
                       any(nd in n for nd in no_decay) and n not in pretrained_weights],
            "weight_decay": 0.0,
            "anneal_w": 0.0,
            "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                any(nd in p_n for nd in no_decay) and p_n not in pretrained_weights]
        }
    ]

    optimizer = RecAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                        anneal_fun=args.recadam_anneal_fun,
                        anneal_k=args.recadam_anneal_k, anneal_t0=args.recadam_anneal_t0,
                        pretrain_cof=args.recadam_pretrain_cof)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    return optimizer, scheduler
