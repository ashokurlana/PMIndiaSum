import datasets
import logging
import os
import sys
import transformers
from dataclasses import dataclass, field
from datasets import load_dataset, load_metric
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          HfArgumentParser, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

check_min_version("4.19.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    my_model_type: str = field(metadata={"help": "Either give mbart50 or indicbartss"})
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    resize_position_embeddings: Optional[bool] = field(default=False, metadata={
        "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."}, )

@dataclass
class DataTrainingArguments:
    train_file: str = field(metadata={"help": "Input training file (a jsonlines or csv file)."})
    validation_file: str = field(metadata={"help": "Input validation file (a jsonlines or csv file)."}, )
    text_column: Optional[str] = field(default="text", metadata={
        "help": "The name of the column in the datasets containing the full texts (for summarization)."}, )
    summary_column: Optional[str] = field(default="summary", metadata={
        "help": "The name of the column in the datasets containing the summaries (for summarization)."}, )
    src_lang_column: Optional[str] = field(default="src_lang", metadata={"help": "the column name for the source language"})
    trg_lang_column: Optional[str] = field(default="trg_lang", metadata={"help": "the column name for the target language"})
    max_source_length: Optional[int] = field(default=1024, metadata={
        "help": "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."}, )
    max_target_length: Optional[int] = field(default=64, metadata={
        "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."}, )

    def __post_init__(self):
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need a train file and a validation file.")
        else:
            assert self.train_file.split(".")[-1] in ["csv", "json"], "`train_file` should be a csv or a json file."
            assert self.validation_file.split(".")[-1] in ["csv", "json"], "`validation_file` should be a csv or a json file."


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    assert model_args.my_model_type in {"indicbartss", "mbart50"}

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)], )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)
    data_files = {"train": data_args.train_file,
                  "validation": data_args.validation_file}
    raw_datasets = load_dataset(data_args.train_file.split(".")[-1], data_files=data_files)
    column_names = raw_datasets["train"].column_names

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)

    if model_args.my_model_type == "indicbartss":
        hard_coded_manipuri_land_id = "<2mni>"
    elif model_args.my_model_type == "mbart50":
        hard_coded_manipuri_land_id = "mni_IN"
    else:
        raise NotImplementedError
    if hard_coded_manipuri_land_id not in tokenizer.vocab.keys():
        temp = len(tokenizer)
        tokenizer.add_tokens([hard_coded_manipuri_land_id])
        assert temp + 1 == len(tokenizer)
        logger.info("Expanding the model embedding with a randomly initialized entry for mni")
        # TODO it would be better to copy "bn" embeddings over
        model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(examples):
        sources = examples[data_args.text_column]
        targets = examples[data_args.summary_column]
        src_langs = examples[data_args.src_lang_column]
        trg_langs = examples[data_args.trg_lang_column]

        if model_args.my_model_type == "indicbartss":
            # https://huggingface.co/ai4bharat/IndicBARTSS
            sources = [source + " </s> " + src_lang for source, src_lang in zip(sources, src_langs)]
            targets = [trg_lang + " " + target + " </s>" for trg_lang, target in zip(trg_langs, targets)]
        elif model_args.my_model_type == "mbart50":
            # https://huggingface.co/docs/transformers/model_doc/mbart#overview-of-mbart50
            sources = [src_lang + " " + source + " </s>" for source, src_lang in zip(sources, src_langs)]
            targets = [trg_lang + " " + target + " </s>" for trg_lang, target in zip(trg_langs, targets)]
        else:
            raise NotImplementedError

        inputs = tokenizer(sources, max_length=data_args.max_source_length, padding=False, truncation=True,
                                 add_special_tokens=False)
        labels = tokenizer(targets, max_length=data_args.max_target_length, padding=False, truncation=True,
                           add_special_tokens=False)
        inputs["labels"] = labels["input_ids"]
        return inputs

    # tokenizer.src_lang = None
    # tokenizer.tgt_lang = None

    train_dataset = raw_datasets["train"]
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=column_names,
                                          desc="Running tokenizer on train dataset", )

    eval_dataset = raw_datasets["validation"]
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=column_names,
                                        desc="Running tokenizer on validation dataset", )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id,
                                           pad_to_multiple_of=8 if training_args.fp16 else None)
    training_args.load_best_model_at_end = True
    training_args.save_total_limit = 2
    training_args.metric_for_best_model = "eval_loss"
    training_args.greater_is_better=False
    trainer = Seq2SeqTrainer(model=model, args=training_args,
                             train_dataset=train_dataset if training_args.do_train else None,
                             eval_dataset=eval_dataset if training_args.do_eval else None, tokenizer=tokenizer,
                             data_collator=data_collator, compute_metrics=None)
    callback = EarlyStoppingCallback(early_stopping_patience=3)
    trainer.add_callback(callback)

    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
