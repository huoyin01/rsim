import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from torch.nn.functional import one_hot, pad
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    HfArgumentParser,
    SchedulerType,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from generate import generate_instructions
from modeling_rism import RsimDecoder, RsimForCausalLM
from riscvTokenizer import riscvTokenizer
from utils import (
    DATASETS_NUM,
    INSTRUCTION_MAX_LENGTH,
    MAX_CONTEXT_SIZE,
    VOCAB_SIZE,
    NoShuffleTrainer
)


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/opt-1.3b",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use the slow or fast tokenizer"},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore mismatched sizes between the model and the tokenizer"
        },
    )

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use."},
    )
    eval_dataset_size: float = field(
        default=0.1,
        metadata={"help": "The proportion of the dataset to use for evaluation."},
    )
    no_keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to remove linebreaks in the text."},
    )
    max_length: int = field(
        default=INSTRUCTION_MAX_LENGTH,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the remote code in datasets loading."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # This class is created to override the default values of the TrainingArguments
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."},
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run eval on the dev set."},
    )
    do_generate: bool = field(
        default=False,
        metadata={"help": "Whether to run instructions generation."},
    )
    num_train_epochs: float = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."},
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    gradient_checkpointing: bool = field(
        default=False
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Adam."},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "A seed for reproducible training."},
    )
    output_dir: str = field(
        default="results",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to adopt."},
    )
    save_total_limit: int = field(
        default=1,
        metadata={
            "help": "Total number of times the model was saved."},
    )
    with_tracking: bool = field(
        default=True,
        metadata={"help": "Whether to log the results to the hub."},
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )
    run_name: str = field(default="none")
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio * total_steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to apply."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for the model."},
    )
    shuffle_on_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the dataset."},
    )
    eval_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to CPU to avoid OOM."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "Whether to return the loss only."},
    )

    
@dataclass
class GenerateArguments:
    do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."},
    )
    penalty_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The values balance the model confidence and the degeneration penalty in contrastive search decoding."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling."},
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={"help": "If set to int > 0, all ngrams of that size can only occur once."},
    )       


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerateArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, generate_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, generate_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(generate_args)
    )
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # download dataset from huggingface
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = {}
        for i in range(1, DATASETS_NUM + 1):
            if i <= 4:
                dataset_name = args.dataset_name + str(i) + "_fix"
            else:
                dataset_name = args.dataset_name + str(i)
            print(f"Downloading dataset {dataset_name}")
            dataset = load_dataset(
                dataset_name, trust_remote_code=args.trust_remote_code
            )
            raw_datasets[dataset_name] = dataset
        raw_datasets = concatenate_datasets([dataset["train"] for dataset in raw_datasets.values()])
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        if "validation" not in raw_datasets.keys() and args.validation_split_percentage > 0:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_CONTEXT_SIZE,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_CONTEXT_SIZE,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer = riscvTokenizer()

    # if args.resume_from_checkpoint:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #         low_cpu_mem_usage=args.low_cpu_mem_usage,
    #         trust_remote_code=args.trust_remote_code,
    #     )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelForCausalLM.from_config(
    #         config,
    #         trust_remote_code=args.trust_remote_code
    #     )
    if args.resume_from_checkpoint:
        model = RsimForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=getattr(args, "low_cpu_mem_usage", False),
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = RsimForCausalLM._from_config(
            config,
            # trust_remote_code=args.trust_remote_code
        )
    model = model.to(torch.float32)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # model.set_decoder(RsimDecoder(model.config))

    """ embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer)) """

    # Tokenize the datasets.
    column_names = raw_datasets.column_names
    instructions_column = "instruction" if "instruction" in column_names else column_names[0]
    coverage_modules_column = "coverage_modules" if "coverage_modules" in column_names else column_names[2]

    def tokenize_function(examples):
        device = torch.device("cpu")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # coverage_modules = torch.zeros(len(COVER_MODULES_RANGE), device=device, dtype=torch.long)
        # for module in examples[coverage_modules_column]:
        #     for name, start, end, index in COVER_MODULES_RANGE:
        #         if start <= module <= end:
        #             coverage_modules[index] += 1
        coverage_modules = torch.tensor(examples[coverage_modules_column], device=device, dtype=torch.float)

        instructions = [str(item) for item in examples[instructions_column]]
        tokenized_inputs = tokenizer(','.join(instructions), padding="max_length", max_length=args.max_length)
        input_ids = torch.tensor(tokenized_inputs["input_ids"], device=device, dtype=torch.long)
        # instruction_labels = one_hot(input_ids, num_classes=VOCAB_SIZE)

        tokenized_inputs["attention_mask"].insert(0, 1)
        attention_mask = torch.tensor(tokenized_inputs["attention_mask"], device=device, dtype=torch.long)
        # attention_mask = einops.repeat(attention_mask, 'n -> (r n)', r=VOCAB_SIZE)

        padding_size = VOCAB_SIZE - coverage_modules.size(0)
        coverage_module_labels = pad(coverage_modules, (0, padding_size), value=0)

        return {
            "input_ids": input_ids,
            "cover_modules": coverage_modules,
            "attention_mask": attention_mask,
            "instruction_labels": input_ids,
            # "instruction_labels": instruction_labels,
            "coverage_module_labels": coverage_module_labels
        }

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=False,
        num_proc=12,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    # DataLoaders creation
    # if "train" in tokenized_datasets:
    # train_dataset = tokenized_datasets
        # Log a few random samples from the training set:
    """
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    """

    """
    if "validation" in tokenized_datasets:
        eval_dataset = tokenized_datasets["validation"]
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator,
            batch_size=args.per_device_eval_batch_size
        )
    else:
        eval_dataloader = None
    """
    if training_args.do_eval:
        total_samples = len(tokenized_datasets)
        split_idx = total_samples - int(total_samples * args.eval_dataset_size)
        train_dataset = tokenized_datasets.select(range(split_idx))
        eval_dataset = tokenized_datasets.select(range(split_idx, total_samples))

        # test metrics
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred

            # 如果模型输出 tuple，只取第一个
            if isinstance(logits, tuple):
                logits = logits[0]

            # 转成 torch 张量并放到 GPU 上（用 float16 可减少显存压力）
            logits = torch.tensor(logits, device="cuda", dtype=torch.float32)
            labels = [torch.tensor(l, device="cuda") for l in labels]

            # 拆分 logits
            instruction_logits = logits[..., :-1, :]
            coverage_module_logits = logits[..., -1, :]

            # GPU 上计算 argmax
            instruction_predictions = torch.argmax(instruction_logits, dim=-1)

            # 拼接（GPU 上直接拼接，不走 Python 循环）
            instruction_predictions = instruction_predictions.reshape(-1)
            coverage_module_logits = coverage_module_logits.reshape(-1)
            instruction_labels = labels[0].reshape(-1)
            coverage_module_labels = labels[1][:, :].reshape(-1)

            # 用 torch 计算准确率（GPU 上）
            ins_accuracy = (instruction_predictions == instruction_labels).float().mean().item()
            cov_accuracy = (coverage_module_logits == coverage_module_labels).float().mean().item()

            # 释放显存
            del logits, labels

            return {
                "ins_accuracy": ins_accuracy,
                "cov_accuracy": cov_accuracy,
            }
    else:
        train_dataset = tokenized_datasets

    os.environ['WANDB_PROJECT'] = 'clm_no_trainer'
    os.environ['WANDB_NAME'] = args.run_name

    # Initialize our Trainer
    trainer = NoShuffleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )

    # print number of trainable parameters
    num_trainable = num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_trainable += p.nelement()
        num_params += p.nelement()
    trainable_perc = num_trainable / num_params
    print(f'{num_trainable=}, {num_params=}, {trainable_perc=:.2%}')

    # print model parameters
    for name, param in model.named_parameters():
        print(f"Param: {name}, dtype: {param.dtype}")
    
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        torch.serialization.add_safe_globals([np.ndarray])
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        # max_train_samples = (
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    if training_args.do_generate:
        file = args.run_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # input_texts = "2,6,2,5,2,2,2,1,24,4,12,1,4,1,1,2,2,1,1,1,1,2"
        input_texts = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        prompt = torch.tensor([int(cov) for cov in input_texts.split(',')], device=device, dtype=torch.long)
        generate_instructions(
            model, 
            tokenizer, 
            prompt, 
            file, 
            max_length=args.max_length,
            do_sample=generate_args.do_sample,
            penalty_alpha=generate_args.penalty_alpha,
            temperature=generate_args.temperature, 
            repetition_penalty=generate_args.repetition_penalty,
            no_repeat_ngram_size=generate_args.no_repeat_ngram_size,
        )


if __name__ == "__main__":
    main()