# Readme
> **This repo provides easy means of inference, training, and evaluation of LLMs** It uses Modal Labs compute to peform 
> its operations.

>IN PROGRESS: Currently working on implementing open source version of this previously private repo.

Currently only these tasks have been confirmed functional:  
1. completion

Todo:  
- [ ] add example data processing for LORA training  
- [ ] fix testing loop to evaluate LORA training
- [ ] fix analyze outputs task
- [ ] fix lora inference script

## Modal install
```
pip install modal-client  
modal token new
```

## Running with modal
FIRST: 

set the MODEL to the model name you'd like. see CACHED_MODELS_MAP for available entries or to add new ones

Run:
```
modal run modal_training.py --task task  

```
e.g. training the llm  

```
modal run modal_training.py --task train_llm  
```

### Available Tasks



## Checkpoints
Ckpts are stored in shared volumes  

## Deploying the script and calling functions

```

import modal  
MODAL_DEPLOYMENT = 'Finetuning_Soleda'  
f = modal.Function.lookup(MODAL_DEPLOYMENT, "make_lora_causal_inference")  
text_response = f.call(input_, model_dir, lora_model_dir)  

```


