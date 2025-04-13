#!/usr/bin/env python

# Link to the chat template used: https://medium.com/@alexandros_chariton/how-to-fine-tune-llama-3-2-instruct-on-your-own-data-a-detailed-guide-e5f522f397d7
# How to quantize properly and run QLORA https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07
# Gained inspiration from https://kaitchup.substack.com/p/llama-2-mt-turn-llama-2-into-a-translation

# Note on adding your own special tokens: https://github.com/huggingface/tokenizers/issues/247

# %%

BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

dataset_dir = "/home/yue.r/train_llama/left_padded"

import argparse

parser = argparse.ArgumentParser(
        prog="llama3.2 distillation script",
        description="distills translation knowledge from llama3.2-3b to llama3.2-1b",
        epilog="Use at your own risk!"
        )

parser.add_argument('-e', '--epochs', help='Number of training epochs', type=int)
parser.add_argument('-lr', '--learning_rate', help='Learning rate for training', type=float)
parser.add_argument('-t', '--teacher_dir', help='The teacher model and adapter location', type=str)
parser.add_argument('-s', '--student_dir', help='The student model and adapter location', type=str)

args = parser.parse_args()

epochs = args.epochs if args.epochs else 3
learning_rate = args.learning_rate if args.learning_rate else 1e-4
teacher_dir = args.teacher_dir if args.teacher_dir else None
student_dir = args.student_dir if args.student_dir else None

print(f"Epochs: {epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Teacher Dir: {teacher_dir} (If 'None', a new adapter will be created)")
print(f"Student Dir: {student_dir} (If 'None', a new adapter will be created)")

from transformers import Trainer, TrainingArguments, GenerationConfig
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
import evaluate

class DistillationTrainingArguments(TrainingArguments):
    # Original vals alpha=0.5, temperature=2.0
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=BATCH_SIZE):
        outputs_stu = model(**inputs)
        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits
        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * loss_fct(
            F.log_softmax(logits_stu / self.args.temperature, dim=-1),
            F.softmax(logits_tea / self.args.temperature, dim=-1))
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
        loss /= (student_training_args.gradient_accumulation_steps * BATCH_SIZE)
        return (loss, outputs_stu) if return_outputs else loss

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
import torch
import os

set_seed(42)

student_ckpt = "meta-llama/Llama-3.2-1B-Instruct"
student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt,
                                                  use_fast=True,
                                                  padding_side="left",
                                                  extra_special_tokens={
                                                      "pad_token":"<|pad|>",
                                                      "end_of_translation_token":"<|end_of_translation|>"
                                                  }
                                                 )

print(f"Tokenization is {student_tokenizer.padding_side} side")

teacher_ckpt = "meta-llama/Llama-3.2-3B-Instruct"

generation_config_teacher = GenerationConfig.from_pretrained(teacher_ckpt)

teacher_model = AutoModelForCausalLM.from_pretrained(teacher_ckpt,
                                             device_map="auto")

teacher_model.resize_token_embeddings(len(student_tokenizer))

teacher_model.generation_config = generation_config_teacher

teacher_model = prepare_model_for_kbit_training(teacher_model)

if not teacher_dir:
    lora_config_teacher = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["down_proj", "up_proj", "gate_proj"]
    )
    
    teacher_peft_model = get_peft_model(teacher_model, lora_config_teacher)
    
else:
    teacher_peft_model = PeftModel.from_pretrained(teacher_model, teacher_dir)
    for param in teacher_peft_model.parameters():
        param.requires_grad = True
    print(f"Teacher adapter at {teacher_dir}")

# %%
from datasets import load_from_disk

# Load the local tokenized dataset

train_tokenized = load_from_disk(f"{dataset_dir}/europarl_dataset/train_tokenized")
val_tokenized = load_from_disk(f"{dataset_dir}/europarl_dataset/val_tokenized")

# %%
student_training_args= DistillationTrainingArguments(
    output_dir=f"./eurollama-distilled-v3-{learning_rate}",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=37,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=epochs,
    fp16=True,
    weight_decay=0.01,
    alpha=1,
    learning_rate=learning_rate,
    max_grad_norm=2,
    report_to=["tensorboard"],
    logging_dir=f"logs-{learning_rate}",
    gradient_accumulation_steps=48,
    warmup_steps=100,
    lr_scheduler_type="linear",
)

# %%
from transformers import AutoConfig

student_config = (AutoConfig.from_pretrained(student_ckpt))

lora_config_student = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["down_proj", "up_proj", "gate_proj"]
)

def student_init():

    generation_config_student = GenerationConfig.from_pretrained(teacher_ckpt)
    stud_unquantized = AutoModelForCausalLM.from_pretrained(student_ckpt,
                                                            config=student_config,
                                                            device_map="auto")

    stud_unquantized.resize_token_embeddings(len(student_tokenizer))

    stud_unquantized.generation_config = generation_config_student
    
    if not student_dir:
        stud_peft_model = get_peft_model(stud_unquantized, lora_config_student)
    else:
        stud_peft_model = PeftModel.from_pretrained(stud_unquantized, student_dir)
        for param in stud_peft_model.parameters():
            param.requires_grad = True
    return stud_peft_model

distillation_trainer = DistillationTrainer(
    model_init=student_init,
    teacher_model=teacher_peft_model,
    args=student_training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=student_tokenizer,
)

distillation_trainer.train()

# %%
distillation_trainer.save_model(f"eurollama-distilled-v3-{learning_rate}")
lora_config_teacher.save_pretrained(f"eurollama-distilled-v3-{learning_rate}/teacher")
teacher_peft_model.save_pretrained(f"eurollama-distilled-v3-{learning_rate}/teacher")

# %%
print("Training done. Model and adapters saved")


