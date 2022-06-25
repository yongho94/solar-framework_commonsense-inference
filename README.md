# solar-framework_commonsense-inference
Code release for "Learning from Missing Relations: Contrastive Learning with Commonsense Knowledge Graphs for Commonsense Inference"


Download Similarity Matrix : 
```bash
gdown https://drive.google.com/uc?id=1EHMIZXP_T1UfSzCWv9Is16n8DRW3dGJx
```

### Preprocessing

#### Download knowledge graphs
Commonsense Knowledge Graph Sources :

* [ConceptNet](https://home.ttic.edu/~kgimpel/commonsense.html)
* [ATOMIC](https://allenai.org/data/atomic)
* [ATOMIC-2020](https://allenai.org/data/atomic-2020)

#### Simple preprocess
* Download the above files, and preprocess each element in tab-separated tsv format.
```
# examples
subject1 \t relation1 \t object1
subject2 \t relation2 \t object2
...
```

* Modify the preprocessed data path in config/{dataset}/dataset.yml
```
name: 'atomic'
truncate:
    subj_len: 25
    obj_len: 25
dir:
    train: {your path}
    dev: {your path}
    test: {your path}
sim: ## <- This is similarity matrix. you can download it from the above url.
    train: {your path}
    dev: {your path}

```

### Fine-tuning

```
python scripts/finetune.py --dataset_type {dataset} --lr {lr} 
```

## Pre-training

```
python scripts/pretrain.py --dataset_type {dataset}
```

