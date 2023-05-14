from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, GPTJForCausalLM

from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict

import gc
import torch
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

MICRO_BATCH_SIZE = 1  # change to 4 for 3090 # originally 8
BATCH_SIZE = 1 # originally 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2  # paper uses 3
LEARNING_RATE = 2e-5  
CUTOFF_LEN = 1500
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b",
                                          add_eos_token=True, 
                                          )

model = GPTJForCausalLM.from_pretrained("databricks/dolly-v1-6b",
                                  load_in_8bit=True,
                                  device_map="auto", 
                                  )

model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

reason_dir = "./gpt3-5_reason_train"
loaded_dataset_dict = DatasetDict.load_from_disk(reason_dir)
data = loaded_dataset_dict["dataset"]

import wandb
wandb.login()
wandb.init(project="dolly_finetuneOnGPT35reason_jupyter")
trainer = transformers.Trainer(
    model=model,
    train_dataset=data, # NOTE: only alpaca data requires indexing into ["train"] here
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        output_dir="lora-dolly-gpt35_01",
        save_total_limit=3,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("6b-lora-dolly-gpt35reasoning01")