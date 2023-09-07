import os
from typing import NewType, Callable, Tuple, List

import torch
import json
from dataclasses import dataclass
import functools

from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

bypass_flag = False


@dataclass
class TrainLog:
    batch: int
    loss: float


def llama2_inputs_processing(model_inputs: List[str]) -> List[str]:
    sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any
    harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    prompt_template = """<s>[INST] <<SYS>>
<<SYS_PROMPT>>
<</SYS>>
<<USER_TEXT>> [/INST]
"""

    # Put sys prompt into prompt template
    prompt_template = prompt_template.replace("<<SYS_PROMPT>>", sys_prompt)

    # put user inputs into prompt template
    processed_model_inputs = [prompt_template.replace('<<USER_TEXT>>', i) for i in model_inputs]

    return processed_model_inputs




def HF_huggingface_model_pipeline(
        model_name: str = 'bert-base-uncased',
        cache_dir='models',
        causal=False
    ) -> Tuple[Callable, PreTrainedModel, PreTrainedTokenizer]:

    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

    # if not Causal -> get output logits
    # if Causal -> generate text

    AutoModelInstance = AutoModel if causal is False else AutoModelForCausalLM
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading Model and Tokenizer
    if 'falcon' in model_name:
        print(f"Falcon has landed :): {model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        language_model = AutoModelInstance.from_pretrained(model_name,
                                               cache_dir=cache_dir,
                                               quantization_config=bnb_config,
                                               torch_dtype=torch.float32,
                                               trust_remote_code=True
                                               )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    elif 'llama-2' in model_name.lower():
        print('llama-2 in the barn :), quant to 4 bit')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        language_model = AutoModelInstance.from_pretrained(model_name,
                                                           cache_dir=cache_dir,
                                                           quantization_config=bnb_config,
                                                           torch_dtype=torch.float32,
                                                           trust_remote_code=True
                                                           )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        # DO NOT SEND BITS AND BYTES MODEL TO DEVICE, THIS IS DONE ALREADY IN QUANTIZATION STEP

    elif model_name == 'google/flan-t5-xl':
        print('loading T5-XL')
        language_model = AutoModelInstance.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        language_model.to(device)
    else:
        print('loading model + tokenizer')
        language_model = AutoModelInstance.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        language_model.to(device)

    print(f"We have a {device}")

    # Create a pipeline
    def pipeline(model_inputs: List[str]):
        """

        :param model_inputs: List of string inputs to pass through the model
        :return:
        """


        # Tokenize
        if model_name == 'google/flan-t5-xl':
            print('Tokenizing FLAN-T5-XL')
            encoded_input = tokenizer(model_inputs, return_tensors='pt', padding='longest')
            input_ids = encoded_input.input_ids.to(device)
            attention_mask = encoded_input.attention_mask.to(device)
            decoder_inputs = tokenizer(['<pad>' + i for i in model_inputs], return_tensors='pt',
                                       padding='longest').input_ids
            decoder_inputs = decoder_inputs.to(device)
        elif 'llama-2' in model_name.lower():
            llama2_processed_inputs = llama2_inputs_processing(model_inputs)
            # print('-'*50)
            # print('Processed Inputs')
            # print('-'*50)
            # print(llama2_processed_inputs)
            # print('-' * 50)
            # print('-' * 50)
            encoded_input = tokenizer(llama2_processed_inputs, return_tensors='pt', padding=True)
            encoded_input.to(device)
        else:
            print('Model name not found. Defaulting to standard tokenization')
            encoded_input = tokenizer(model_inputs, return_tensors='pt', padding=True)
            encoded_input.to(device)

        get_GPU_usage('after passing inputs to GPU')

        # Forward Pass
        if 'falcon' in model_name or 'llama-2' in model_name.lower():
            print('Forward Pass')

            if causal is False:
                output = language_model(input_ids=encoded_input['input_ids'],
                                        attention_mask=encoded_input['attention_mask'])
                print(output.last_hidden_state.dtype)
            else:
                print('Generate text')
                try:
                    print(encoded_input.keys())
                    print(tokenizer.model_max_length)
                except:
                    print('Couldnt print keys')
                    pass

                output = language_model.generate(
                    input_ids=encoded_input['input_ids'],
                    attention_mask=encoded_input['attention_mask'],
                    max_new_tokens=None,
                    temperature=0.3
                )

        elif model_name == 'google/flan-t5-xl':
            print('Forward Pass Flan-t5-xl')
            output = language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_inputs
            )
        else:
            output = language_model(**encoded_input)

        print('returning language model output')
        return output

    return pipeline, language_model, tokenizer



def dump_logs(logs, dir_name):
    fp_log = os.path.join(dir_name, f'runlog.json')
    with open(fp_log, 'w') as f:
        json.dump(logs, f, indent=4)


def bypass_if_true(flag_variable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if flag_variable:
                pass
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


@bypass_if_true(bypass_flag)
def get_GPU_usage(prefix: str = ''):
    prefix = prefix + '_' if prefix != '' else prefix
    gpu_memory = torch.cuda.memory_allocated()
    print(prefix + f"Memory allocated on GPU: {gpu_memory / 1024 ** 3:.2f} GB")


def print_directory_contents(path, indent_level=0):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            print("|  " * indent_level + "|--" + file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            print("|  " * indent_level + "|--" + dir_path)
            print_directory_contents(dir_path, indent_level + 1)

SharedVolumePath = NewType('SharedVolumePath', str)
