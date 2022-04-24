from torch.nn import Linear
from transformers import BartForConditionalGeneration

def convert_model(src_model, dst_model):
    params_src = src_model.named_parameters()
    params_dst = dst_model.named_parameters()
    dict_src = dict(params_src)
    dict_dst = dict(params_dst)
    match_fail = 0
    for name in dict_src:
        dst_name = f'model.{name}'
        if dst_name in dict_dst:
            dict_dst[dst_name].data.copy_(dict_src[name].data)
        elif name in dict_dst:
            dict_dst[name].data.copy_(dict_src[name].data)
        else:
            match_fail += 1
            print(f'Unmatched layer of dst_model to src_model : layer name : {dst_name}')

    if match_fail == 0:
        print('All layered are matched')

    return dst_model

