import math
import shutil
import sys
import warnings
import time
from pathlib import Path
from typing import List, Dict

from packaging import version
import torch
from tqdm import tqdm
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainerCallback, \
    AutoTokenizer, TrainerState, AutoModelForSeq2SeqLM, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig, TaskType
import os
import json

from transformers.debug_utils import DebugOption
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.integrations import hp_params
from transformers.pytorch_utils import is_torch_less_than_1_11
from transformers.trainer import TRAINER_STATE_NAME

from src.utils.utils import get_GPU_usage
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    get_model_param_count,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode
from transformers.utils import (
    logging,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available, is_accelerate_available,
)


class InferenceUtteranceDataset(Dataset):
    def __init__(self, dataset):
        # Load the JSON data
        self.data = dataset.to_list()

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Get a specific sample from the dataset at the given index
        sample = self.data[index]
        utterance = sample['utterance']

        # Concatenate the utterance with "<sep>"
        input_ = f"{utterance} <sep>"

        return input_, sample['intent'], sample['dact']

    def load_data(self, json_file):
        # Replace this line with your own JSON loading logic
        loaded_data = json.load(open('utterances.json', 'r'))

        return loaded_data


def formatting_function_utterances(example, tokenizer):
    """Prepare text from json utterances, intents, and dacts"""
    return f"{example['utterance']} {tokenizer.sep_token} {example['intent']} <label> {example['dact']}"


def get_custom_tokenizer(tokenizer):
    """
    Create custom tokens for the tokenizer.

    :param tokenizer:
    :return: tokenizer with modified special tokens
    """
    existing_special_tokens = tokenizer.additional_special_tokens
    custom_tokens = ['<header>']
    existing_special_tokens.extend(custom_tokens)
    special_tokens = {
        'additional_special_tokens': existing_special_tokens,
        # 'sep_token': '<sep>',
        # 'pad_token': '<pad>'
    }
    tokenizer.add_special_tokens(special_tokens)

    print('Adding tokens to tokenizer.')
    print(tokenizer.additional_special_tokens)
    print(tokenizer.pad_token, tokenizer.sep_token)

    return tokenizer


def prepare_causal_data_and_tokenizer(fp: str, tokenizer: PreTrainedTokenizer, train_size:float = 0.9):
    """

    :param train_size:
    :param fp: filepath to retrieve text data from
    :param tokenizer: tokenizer instance
    :return:
    """


    from datasets import load_dataset
    from trl.trainer import ConstantLengthDataset

    # for debugging show filepath and current directory contents
    print('-'*50)
    print(fp)
    print(os.listdir())
    print(os.listdir('volume'))
    print('-'*50)

    # HF load dataset util
    data = load_dataset('text', data_dir=fp)

    if train_size < 1:
        data = data['train'].train_test_split(train_size=0.9, shuffle=True, seed=15)

    train_data = data['train']
    test_data = data['test'] if train_size < 1 else None

    # set max length of sample for passing to the model
    # this will be an issue depending on the model. Here is is set to a max of 2048 because model_max_length
    # is incorrect in the Falcon models
    max_length = min(2048, tokenizer.model_max_length)

    # Grab a constant length dataset that will fill the entire context window of the LLM while training. Maximizing each
    # forward pass
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        dataset_text_field='text',
        formatting_func=None,  # used if you need to manipulate the text inside the
        infinite=True,
        seq_length=max_length,
    )

    test_dataset = ConstantLengthDataset(
        tokenizer,
        test_data,
        dataset_text_field='text',
        formatting_func=None,  # used if you need to manipulate the text inside the
        infinite=True,
        seq_length=max_length,
    )

    print(f'Train dataset has: {len(train_dataset)}')
    print(f'Test dataset has: {len(test_dataset)}')

    return train_dataset, test_dataset


def tokenization_seq2seq(examples: Dict, tokenizer: PreTrainedTokenizer):
    """

    :param examples: dictionary with keys [text, label_1, label_2]
    :param tokenizer:
    :return:

    """
    samples = [example + tokenizer.sep_token for example in examples['text']]
    labels = [intent + str('<label>') + dact for intent, dact in zip(examples['label_1'], examples['label_2'])]
    outputs = tokenizer(samples) # grab tokenizer dict output
    labels = tokenizer(labels)['input_ids']
    return {'labels': labels} | outputs


def prepare_seq2seq_data_and_tokenizer(fp, model, tokenizer):
    """
    seq2seq models with encoder-decoder architecture need both encoder and decoder input ids. When training with next
    word prediction we only need to offset the labels to the decoder one-to-the-right to generate the decoder inputs.
    The DataCollatorForSeq2Seq pads inputs to the appropriate sequence length and will prepare the decoder ids based on
    input labels.

    calculate the loss on the next token prediction. This means that the
    :param fp: location of data
    :param model:
    :param tokenizer:
    :return:
    """
    from datasets import load_dataset
    from transformers.data import DataCollatorForSeq2Seq

    data = load_dataset('json', data_files=fp)
    data = data['train'].train_test_split(train_size=0.9, shuffle=True, seed=15)
    train_dataset, validation_dataset = data['train'], data['test']

    train_dataset = train_dataset.map(
        lambda x: tokenization_seq2seq(x, tokenizer),
        batched=True
    )
    validation_dataset = validation_dataset.map(
        lambda x: tokenization_seq2seq(x, tokenizer),
        batched=True
    )

    return train_dataset, validation_dataset, DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer,
                                                                     padding='longest')


def best_checkpoint_lora(lora_model_dir: str):
    """
    Find the most recent ckpt in dir
    :param lora_model_dir:
    :return:
    """
    dirs = [d for d in os.listdir(lora_model_dir) if
            os.path.isdir(os.path.join(lora_model_dir, d))
            and 'checkpoint-' in d]

    latest_ckpt = 'checkpoint-' + str(max([int(dir.split('-')[-1]) for dir in dirs]))
    best_ckpt_path = os.path.join(lora_model_dir, latest_ckpt)

    while 'trainer_state.json' not in os.listdir(best_ckpt_path):
        latest_ckpt = latest_ckpt.split('-')[0] + '-' + str(int(latest_ckpt.split('-')[1]) - 10)
        best_ckpt_path = os.path.join(lora_model_dir, latest_ckpt)

    print(f'Picking {latest_ckpt}')

    return latest_ckpt


def lora_inference(inputs: List[str],
                   lora_model_dir: str,
                   language_model_dir: str,
                   quantization: int,
                   model_type: str):
    """

    Args:
        inputs: Strings to pass through model
        lora_model_dir: LORA CKPT directory
        language_model_dir: cache_dir where model is stored. Will only be downloaded first time ran.
        quantization: 4, 8
        model_type: causal or seq2seq

    Returns:

    """
    best_ckpt_dir = os.path.join(lora_model_dir, best_checkpoint_lora(lora_model_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # grab tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(best_ckpt_dir,
                                              trust_remote_code=True
                                              )

    if quantization == 4:
        print('Loading in 4 bit')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif quantization == 8:
        print('Loading in 8 bit')
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        raise ValueError('Provide a valid quantization parameter (4 or 8)')

    print("Loading the model")
    config = PeftConfig.from_pretrained(best_ckpt_dir)

    match model_type:
        case 'causal':
            language_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                cache_dir=language_model_dir,
                quantization_config=bnb_config,
                return_dict=True,
                device_map='auto',
                trust_remote_code=True,
            )
        case 'seq2seq':
            language_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path,
                cache_dir=language_model_dir,
                quantization_config=bnb_config,
                return_dict=True,
                device_map='auto',
                trust_remote_code=True,
            )

    language_model.config.use_cache = False
    language_model.resize_token_embeddings(len(tokenizer))

    head_weights = Path(lora_model_dir + "/lm_head_state_dict.pth")
    if head_weights.is_file():
        lm_head_weights = torch.load(os.path.join(lora_model_dir, "lm_head_state_dict.pth"))
        head = language_model.get_output_embeddings()
        head.load_state_dict(lm_head_weights)

    model = PeftModel.from_pretrained(language_model, best_ckpt_dir)

    model.eval()
    model.base_model.eval()

    with torch.no_grad():
        X = inputs
        encoded_input = tokenizer(X, return_tensors='pt', padding=True)
        encoded_input.to(device)
        output = model.generate(
            input_ids=encoded_input['input_ids'],
            attention_mask=encoded_input['attention_mask'],
            max_new_tokens=300,
            temperature=0,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    text_response = tokenizer.batch_decode(output)
    return text_response


def run_training(
        fp: str,
        model_name: str,
        model_folder: str,
        output_dir: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        quantization: int,
        model_type: str = 'causal',
        new_tokens: bool = False
):
    """

    Args:
        fp: file path to data
        model_name: must be hugging face name such as tiiuae/falcon-7b or google/flan-t5-xxl
        model_folder: cache_dir for models
        output_dir: to save ckpt files
        epochs: number of epochs
        batch_size:
        learning_rate: for the optimizer
        quantization: 4 or 8 quantization
        model_type: causal or seq2seq
        :param new_tokens:
    """

    # Quantization
    if quantization == 4:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif quantization == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        raise ValueError('Provide a valid quantization parameter (4 or 8)')
    print(f'Loaded in quantization: {quantization}')

    # grab model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=model_folder,
                                              trust_remote_code=True)
    print(f'Model has a max length of : {tokenizer.model_max_length}')

    # Grab custom tokenizer if needed.
    if new_tokens:
        tokenizer = get_custom_tokenizer(tokenizer)

    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)

    # load model and prepare data based on model_type
    data_collator = None
    if model_type == 'causal':

        train_dataset, validation_dataset = prepare_causal_data_and_tokenizer(
            fp,
            tokenizer=tokenizer,
            train_size=1 # use all the data if you're just trying to learn content
        )

        print("Loading the model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=model_folder,
            quantization_config=bnb_config, # load with defined BnB config
            device_map="auto",
            trust_remote_code=True
        )

    elif model_type == 'seq2seq':
        print("Loading the model")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=model_folder,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # seq2seq specific formatting of val and train data
        train_dataset, validation_dataset, data_collator = prepare_seq2seq_data_and_tokenizer(
            fp=fp, model=model, tokenizer=tokenizer
        )
    get_GPU_usage('after loading model')

    # Need to resize token embeddings if we have changed the vocab size
    if new_tokens:
        model.resize_token_embeddings(vocab_size)

    model.config.use_cache = False

    # check for older LORA ckpt of training.
    resume_ckpt = True if os.listdir(output_dir) and 'checkpoint' in ''.join(os.listdir(output_dir)) else False
    best_ckpt_dir = None
    if resume_ckpt:
        best_ckpt_dir = os.path.join(output_dir, best_checkpoint_lora(output_dir))
        print(f'Best checkpoint found at: {best_ckpt_dir}')

        # Load model head if its been resized
        if new_tokens:
            """
                When finetuning with new tokens, you have to add to the vocabulary. This requires that you expand the
                output head. This also means that you have to load the custom head from disk in the case of a ckpt.
            """
            lm_head_save_path = os.path.join(output_dir, "lm_head_state_dict.pth")
            model.get_output_embeddings().load_state_dict(torch.load(
                lm_head_save_path
            ))

        # load tokenizer
        tknzer = AutoTokenizer.from_pretrained(best_ckpt_dir, padding_side='left')

        # Update data processing with ckpt tokenizer
        match model_type:
            case 'causal':
                train_dataset.tokenizer = tknzer
                validation_dataset.tokenizer = tknzer
            case 'seq2seq':
                data_collator.tokenizer = tknzer

    # set model dependent lora config parameters
    match model_type:
        case 'causal':
            task = TaskType.CAUSAL_LM
            target_modules = [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]
        case 'seq2seq':
            task = TaskType.SEQ_2_SEQ_LM
            target_modules = ['q', 'v']
        case other:
            ValueError(f'Incorrect task passed. Passed {other}')

    # define LORA parameters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=task

    )

    # prep layers for k bit training
    # freezes base model layers
    model = prepare_model_for_kbit_training(model)

    # grab PEFT model
    if resume_ckpt:
        print('Loading PEFT model from ckpt')
        model = PeftModel.from_pretrained(model, best_ckpt_dir, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config=lora_config)

    # Unfreeze lm_head if we need to train for new tokens
    # PEFT Methods freeze it
    if new_tokens:
        head = model.get_output_embeddings()

        for param in head.parameters():
            param.requires_grad = True
            print(f'After setting: {param.requires_grad}')

    # Training and args
    print("Starting main loop")
    len_dataloader = int(np.ceil(len(train_dataset) / batch_size))
    gradient_accumulation_steps = 1
    num_update_steps_per_epoch = len_dataloader // gradient_accumulation_steps
    print(f'Expected steps per epoch: {num_update_steps_per_epoch}')

    logging_directory = Path('output_dir/logs')
    logging_directory.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        # evaluation_strategy="epoch",
        num_train_epochs=epochs,
        save_strategy="steps",
        save_steps=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        run_name="finetuning_" + model_name.split('/')[-1],
        optim="paged_adamw_8bit",
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir=str(logging_directory),
        logging_strategy='steps',
        logging_steps=25,
        fp16=True,

    )

    callbacks = [PeftSavingCallback]
    CurrentTrainer = MyTrainer  # if best_ckpt_dir is not None else Trainer

    trainer = CurrentTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        args=training_args,
        data_collator=data_collator

    )

    print("Training...")
    print(f'We are training: {model_name}')
    print(f'Adapter model is on: {model.device}')
    print(f'Language model is on: {model.base_model.device}')
    print(f'Trainable parameters: {[i for i, p in enumerate(model.parameters()) if p.requires_grad][-1] + 1}')

    if best_ckpt_dir is not None:
        print('Resuming from checkpoint')
        trainer.train(resume_from_checkpoint=best_ckpt_dir)
    else:
        print('Training anew')
        trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(output_dir, "lora_final_checkpoint/"))

    if new_tokens:
        head = trainer.model.base_model.get_output_embeddings()
        print(head.state_dict())
        torch.save(head.state_dict(), lm_head_save_path)


class PeftSavingCallback(TrainerCallback):

    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)

            # base_model_save_path = os.path.join(args.output_dir, "lm_head_state_dict.pth")

            # head = kwargs["model"].base_model.get_output_embeddings()
            # torch.save(head.state_dict(), base_model_save_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


logging.set_verbosity_debug()
logger = logging.get_logger(__name__)

skip_first_batches = None
if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    if version.parse(accelerate_version) >= version.parse("0.16"):
        from accelerate import skip_first_batches


# Custom trainer class that allows you to resume ckpt based from a lora ckpt
class MyTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(
            self,
            resume_from_checkpoint=None,
            trial=None,
            ignore_keys_for_eval=None,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True
            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # should this be under the accumulate context manager?
                # the `or` condition of `steps_in_epoch <= args.gradient_accumulation_steps` is not covered
                # in accelerate
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
