import json
from pathlib import Path
from typing import Optional, List, Dict, Union

import modal
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from modal import Image, SharedVolume, Stub
import os
from src.utils.llm_finetuning_utils import run_training, lora_inference, \
    InferenceUtteranceDataset
from src.utils.metrics import analyze_outputs
from src.utils.utils import SharedVolumePath, HF_huggingface_model_pipeline, get_GPU_usage, print_directory_contents

CACHED_MODELS_MAP = {
    'falcon-7b': ('tiiuae/falcon-7b', 'causal'),
    'koala-7b': ('TheBloke/koala-7B-HF', 'causal'),
    'flan-xl': ('google/flan-t5-xl', 'seq2seq'),
    'falcon-40b': ('tiiuae/falcon-40b', 'causal'),
    'dolly-7b': ('databricks/dolly-v2-7b', 'causal'),
    'llama-2-13b-chat-hf': ('meta-llama/Llama-2-13b-chat-hf', 'causal')
}

MODEL = 'llama-2-13b-chat-hf'
MODEL_TYPE = CACHED_MODELS_MAP[MODEL][1]
MODEL_NAME = CACHED_MODELS_MAP[MODEL][0]


MODAL_DEPLOYMENT = 'POC_LLM_FINETUNING'
MODAL_VOLUME = 'POC_LLMS_VOLUME'
stub = Stub(MODAL_DEPLOYMENT)

# env var

stub["HF_TOKEN"] = modal.Secret.from_dict({'HF_TOKEN': os.environ["HF_TOKEN"]})


# Set up remote directory structure
volume = SharedVolume().persist(MODAL_VOLUME)
MODEL_FOLDER = SharedVolumePath('models')  # where in root to place model cache
CACHE_PATH = Path('/root/volume')  # where in root to place remove volume
MODEL_FOLDER = CACHE_PATH / MODEL_FOLDER

DATA_FOLDER = CACHE_PATH / 'coppermind_wiki'
SOURCE_TEXTS = SharedVolumePath(str(DATA_FOLDER)) # directory where data is kept for model training

def download_now():
    from huggingface_hub import snapshot_download

    cache_dir = MODEL_FOLDER

    snapshot_download(MODEL_NAME,
                      cache_dir=cache_dir)


llm_image = Image.micromamba(
    python_version='3.10',
    force_build=False) \
    .micromamba_install(
    "cudnn=8.1.0",
    "cuda-nvcc",
    channels=["conda-forge", "nvidia"],
) \
    .apt_install(["git"]) \
    .pip_install(
    'bitsandbytes',
    'datasets',
    'trl',
    "huggingface_hub",
    "transformers",
    "peft",
    "accelerate",
    "einops==0.6.1",
    "torch",
    "sentence-transformers",
    "scikit-learn", 'accelerate',
    "pytz"
).run_function(download_now)


@stub.local_entrypoint()
def main(task):
    match task:
        case 'completion':  # basic causal inference on pretrained model from checkpoint
            input_ = 'Explain regularization for deep neural networks in detail.\n'
            text_response = make_causal_inference.call(input_=input_, model_name=MODEL_NAME)
            print(text_response)
        # Lora training, testing, analysis, and basic inference
        case 'train_llm':
            train_llm.call()
        case 'test_loop_lora':
            # lora model checkpoint directory
            lora_model_dir = SharedVolumePath('volume/lora_tuning/falcon-7b')

            test_run_name = MODEL

            # Grab model predictions
            save_dict = test_loop_lora.call(
                model_dir=MODEL_FOLDER,
                lora_model_dir=lora_model_dir,
                batch_size=200,
                model_type=MODEL_TYPE,
                fp='utterances.json'
            )

            save_path = os.path.join('data', test_run_name, 'predictions.json')
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            json.dump(save_dict, open(save_path, 'w'), indent=4)

        case 'analyze_outputs':
            # this is a local function. DOES NOT GO TO MODAL
            # THIS FILE PATH IS A LOCAL FILEPATH
            predictions_file_path = 'data/flan-xl_8bit/predictions.json'
            analyze_outputs(predictions_file=predictions_file_path,
                            model_type=MODEL_TYPE)

        case 'lora_inference':  # basic causal inference from a LORA checkpoint
            input_ = 'Whats the weather today? <sep>'
            lora_model_dir = SharedVolumePath('volume/lora_tuning_3/falcon-7b_8bit')  # shared volume path

            text_response = make_lora_inference.call(input_=input_,
                                                     model_dir=MODEL_FOLDER,
                                                     lora_model_dir=lora_model_dir)
            print(text_response)

        case 'clean_volume':
            # clean up utility for modal volumes
            remove_dirs = ['']
            keep_dirs = ['models', 'runs', 'utterances.json']
            directory_to_clean = 'volume'
            clean_dirs.call(remove_dirs=None,
                            keep_dirs=keep_dirs,
                            current_directory=directory_to_clean,
                            delete=True)


@stub.function(
    gpu='A100',
    cloud='gcp',
    cpu=4,
    memory=16000,
    image=llm_image,
    shared_volumes={CACHE_PATH: volume},
    timeout=5400
)
def train_llm(fp=None):
    # path to data
    if fp is None:
        fp = SOURCE_TEXTS

    # model and quantization
    quantization = 4

    # ckpt directory
    output_dir = os.path.join('volume/lora_tuning/',
                              MODEL + f'_{quantization}bit')

    os.makedirs(output_dir, exist_ok=True)
    run_training(
        fp=fp,
        model_name=MODEL_NAME,
        model_folder=MODEL_FOLDER,
        output_dir=output_dir,
        epochs=1,
        batch_size=8,
        learning_rate=2e-4,
        quantization=quantization,
        model_type=MODEL_TYPE
    )


@stub.function(
    gpu="A100",
    cpu=2,
    cloud='gcp',
    memory=32768,
    image=llm_image,
    shared_volumes={CACHE_PATH: volume},
    timeout=2700,
    secret=stub["HF_TOKEN"]
)
def make_causal_inference(input_: Union[List[str], str], model_name: str):
    from huggingface_hub import login
    login(token=os.environ['HF_TOKEN'])

    inputs = input_ if isinstance(input_, List) else [input_]
    print('saving to', MODEL_FOLDER)

    # Construct a pipeline that handles tokenization and passing text through the model
    language_model_pipeline, language_model, tokenizer = HF_huggingface_model_pipeline(
        model_name=model_name,
        cache_dir=MODEL_FOLDER,
        causal=True
    )

    with torch.no_grad():
        X = inputs
        language_model_outputs = language_model_pipeline(X)

    get_GPU_usage()
    text_response = tokenizer.batch_decode(language_model_outputs)

    return text_response


@stub.function(
    gpu='A100',
    cloud='gcp',
    memory=32768,
    image=llm_image,
    shared_volumes={CACHE_PATH: volume},
    timeout=5400
)
def make_lora_inference(input_: Union[List[str], str],
                        model_dir: str,
                        lora_model_dir: str):
    print([os.path.join(lora_model_dir, p) for p in os.listdir(lora_model_dir)])

    # quantization
    quantization = int(lora_model_dir.split('_')[-1][0])
    if quantization in (4, 8):
        print(f'{quantization} quantization')
    else:
        print(f'Quantization parameter is {quantization}')
        raise ValueError('Quantization parameter must be 4 or 8')

    inputs = input_ if isinstance(input_, List) else [input_]
    text_response = lora_inference(inputs=inputs,
                                   lora_model_dir=lora_model_dir,
                                   language_model_dir=model_dir,
                                   quantization=quantization,
                                   model_type=MODEL_TYPE)
    print(text_response)
    return text_response


@stub.function(
    gpu='A100',
    cloud='gcp',
    cpu=4,
    memory=32768,
    image=llm_image,
    shared_volumes={CACHE_PATH: volume},
    timeout=2700
)
def test_loop_lora(
        model_dir: str,
        lora_model_dir: str,
        batch_size: int,
        model_type: str,
        fp: str = 'data/utterances.json'
):
    """

    :param model_dir: directory where model is cached to prevent redownloading
    :param lora_model_dir: directory of lora checkpoint
    :param batch_size:
    :param model_type: causal or seq2seq
    :param fp: path to dataset being processed
    :return:
    """
    raise NotImplementedError
    # grab data from json file
    data = load_dataset('json', data_files=fp)
    data = data['train'].train_test_split(train_size=0.9, shuffle=True, seed=15)
    train_dataset, validation_dataset = data['train'], data['test']

    # Create an instance of the custom dataset
    dataset = InferenceUtteranceDataset(validation_dataset)

    # Create a data loader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # quantization
    quantization = int(lora_model_dir.split('_')[-1][0])
    if quantization in (4, 8):
        print(f'{quantization} quantization')
    else:
        print(f'Quantization parameter is {quantization}')
        raise ValueError('Quantization parameter must be 4 or 8')

    # Iterate over the data loader to get the samples
    inputs_save = []
    target_intents = []
    target_dacts = []
    predictions = []
    ii = 0
    print('About to start testing loop')
    for input_, intents, dacts in data_loader:
        print(ii)
        ii += 1

        target_intents.extend(intents)
        target_dacts.extend([split_if_plus(d) for d in dacts])
        inputs = list(input_)
        text_response = lora_inference(inputs=inputs,
                                       lora_model_dir=lora_model_dir,
                                       language_model_dir=model_dir,
                                       quantization=quantization,
                                       model_type=model_type)
        predictions.extend(text_response)
        inputs_save.extend(input_)

    print(target_dacts, target_intents, inputs)
    text_response = ''
    print(text_response)
    save_dict = {
        'target_intents': target_intents,
        'target_dacts': target_dacts,
        'predictions': predictions,
        'inputs': inputs_save
    }
    return save_dict


@stub.function(
    image=llm_image,
    shared_volumes={CACHE_PATH: volume},
    allow_cross_region_volumes=True)
def clean_dirs(current_directory: str = 'volume/runs',
               keep_dirs: List[str] = None,
               remove_dirs: Optional[List[str]] = None,
               delete=False):
    import shutil
    print(f'In: {os.getcwd()}')
    print(os.listdir(current_directory))
    dirs = set(os.listdir(current_directory))

    if keep_dirs is None and remove_dirs is None:
        raise ValueError('Must pass directories to remove')
    if remove_dirs and keep_dirs:
        raise ValueError('Pass only keep_dirs OR remove_dirs, NOT both')
    if remove_dirs:
        dirs = set(os.listdir(current_directory))
        keep_dirs = dirs - set(remove_dirs)

    keep_dirs = dirs & set(keep_dirs)

    print(f'Keeping: {keep_dirs}')
    print(f'Removing: {dirs - keep_dirs}')

    if delete:
        # Iterate through the directories in the current directory
        for item in os.listdir(current_directory):
            item_path = os.path.join(current_directory, item)
            if os.path.isdir(item_path) and item not in keep_dirs:
                # Check if the directory is empty
                if not os.listdir(item_path):
                    # Delete the directory if it's empty
                    os.rmdir(item_path)
                else:
                    # Delete the directory and its contents if it's not empty
                    shutil.rmtree(item_path)

    print(os.listdir(current_directory))
