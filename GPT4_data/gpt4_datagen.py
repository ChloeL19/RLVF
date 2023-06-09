from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed
)  # for exponential backoff

import json

import requests
import re

import yaml

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

openai.organization = "org-f2tK1brD8eM1W91o2X5WgNoy"
openai.api_key = config['OPENAI_KEY']

def coqc(v):
  '''
  Returns line number of first error. -2 iff no errors, and -1 otherwise.
  '''
  r = requests.post("https://coq.livecode.ch/check", data = { 'v': v }).json()
  r.raise_for_status()
  if r['status'] == 0:
    return -2
  r = r['log']

  pattern = r'line (\d+),'
  match = re.search(pattern, r)  
  if match:
    line_number = int(match.group(1))
  else:
    line_number = -1
  return line_number

def cfeedback(v):
  '''
  Returns the compiler error if one exists. Returns None if everything compiles cleanly.
  '''
  r = requests.post("https://coq.livecode.ch/check", data = { 'v': v }).json()
  if r['status'] == 0:
    return None
  r = r['log']
  return r

def get_linenumber(cf):
  pattern = r'line (\d+),'
  match = re.search(pattern, cf)  
  if match:
    line_number = int(match.group(1))
  else:
    line_number = -1
  return line_number

def get_totallines(response):
    return len(response.split('\n'))

def get_line(line_number, response):
    broken = response.split('\n')
    return broken[line_number-1]

config = PPOConfig(
    model_name="edbeeching/gpt-neo-125M-imdb-lora-adapter-merged",
    learning_rate=1.41e-5,
    log_with='wandb',
    mini_batch_size=1,# prev: 16
    batch_size=1, # prev: 256, but working with super limited samples so will try lower batch size for now
    # gradient_accumulation_steps=1, --> apparently this is unrecognized
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config.mini_batch_size}

def build_dataset(config, dataset_name="../MBPP dataset/MBPP_Coq_Train.csv"):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.
    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.
    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("csv", data_files=dataset_name, split="train")

    def concat(sample):
      ex = sample['specification'] + "Test case 1: " + sample['test_case_1'] + \
      ", test case 2: " + sample['test_case_2'] + ", test case 3: " + sample['test_case_3']
      return ex

      # return sample['specification'] + "Test case 1: " + sample['test_case_1'] + \
      # ", test case 2: " + sample['test_case_2'] + ", test case 3: " + sample['test_case_3'] + " Prove some formal properties. Please only write code for the last stand-alone example. *)"


    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(concat(sample))
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

# multi-shot boilerplate
multishot = "(* Stand-alone Example 1: Write a function that doubles a number. Test case 1: double 3 = 6. Prove some formal properties. *) \nFixpoint double (n: nat): nat := match n with | 0 => 0 | S n => S (S (double n)) end. \n\nLemma example_double_3: double 3 = 6.\nProof. simpl. reflexivity. Qed. \n\n Theorem theorem_double_distribute: \nforall a b, double a + double b = double (a + b).\n Proof.\n intros.\n induction a.\n - simpl. reflexivity.\n - simpl. rewrite IHa. reflexivity. \n Qed. \n\n (* Stand-alone Example 2: Write a function that creates a list of n elements. Test case 1: replicate 1 0 = []. Test case 2: replicate 1 2 = [1; 1]. Prove some formal properties. *) \n Require Import Coq.Lists.List. \n Open Scope list_scope. \n Import ListNotations. \n Fixpoint replicate {X: Type} (x: X) (n: nat): list X := \n match n with \n | 0 => []\n | S n => x :: replicate x n \n end. \n Lemma example_replicate_0: replicate 1 0 = []. \n Proof. simpl. reflexivity. Qed.\n Lemma example_replicate_2: replicate 1 2 = [1; 1].\n Proof. simpl. reflexivity. Qed.\n\n Theorem replicate_length:\n\t forall n, length (replicate 1 n) = n.\n Proof. \n intros. \n induction n.\n - simpl. reflexivity. \n - simpl. rewrite IHn. reflexivity.\n Qed. \n Theorem replicate_length_any: \n\t forall (X: Type) (x: X) n, length (replicate x n) = n. \n Proof.\n intros. \n induction n.\n - simpl. reflexivity.\n- simpl. rewrite IHn. reflexivity.\n Qed."

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config)

systemText = """ You are an AI assistant helping users write Coq code in order to implement given function specifications. 
1. The program you write should only contain Coq code in response to the given function specification. 
3. Any step-by-step reasoning that is not Coq code should be written as a comment.
3. As the user provides compiler feedback, modify and update the program accordingly and keep the variables and the general program structure consistent.
4. In addition to implementing the function, give at least 2 properties as theorems with their proofs.

The following are examples.

Query from user:
Write a function that creates a list of n elements. Test case 1: replicate 1 0 = []. Test case 2: replicate 1 2 = [1; 1]. 

Response from assistant:
\`\`\`
Require Import Coq.Lists.List. 
 Open Scope list_scope. 
 Import ListNotations. 
 Fixpoint replicate (x: X) (n: nat): list X := 
 match n with 
 | 0 => []
 | S n => x :: replicate x n 
 end. 
 Lemma example_replicate_0: replicate 1 0 = []. 
 Proof. simpl. reflexivity. Qed.
 Lemma example_replicate_2: replicate 1 2 = [1; 1].
 Proof. simpl. reflexivity. Qed.

 Theorem replicate_length:
	 forall n, length (replicate 1 n) = n.
 Proof. 
 intros. 
 induction n.
 - simpl. reflexivity. 
 - simpl. rewrite IHn. reflexivity.
 Qed. 
 Theorem replicate_length_any: 
	 forall (X: Type) (x: X) n, length (replicate x n) = n. 
 Proof.
 intros. 
 induction n.
 - simpl. reflexivity.
- simpl. rewrite IHn. reflexivity.
 Qed.
\`\`\`

Query from user:
Your code produces an error in the line Fixpoint replicate (x: X) (n: nat): list X :=\n{}Can you please explain what this error means? Let's think step by step. Please rewrite all code if you rewrite any code.
File \"./ex.v\", line 4, characters 24-25:\nError: The reference X was not found in the current environment.

Response from assistant:
\`\`\`
Require Import Coq.Lists.List. 
 Open Scope list_scope. 
 Import ListNotations. 
 Fixpoint replicate '{X: Type'} (x: X) (n: nat): list X := 
 match n with 
 | 0 => []
 | S n => x :: replicate x n 
 end. 
 Lemma example_replicate_0: replicate 1 0 = []. 
 Proof. simpl. reflexivity. Qed.
 Lemma example_replicate_2: replicate 1 2 = [1; 1].
 Proof. simpl. reflexivity. Qed.

 Theorem replicate_length:
	 forall n, length (replicate 1 n) = n.
 Proof. 
 intros. 
 induction n.
 - simpl. reflexivity. 
 - simpl. rewrite IHn. reflexivity.
 Qed. 
 Theorem replicate_length_any: 
	 forall (X: Type) (x: X) n, length (replicate x n) = n. 
 Proof.
 intros. 
 induction n.
 - simpl. reflexivity.
- simpl. rewrite IHn. reflexivity.
 Qed.
\`\`\`"""

messages=[{"role": "system", "content": systemText}]

# TODO: use the ConversationBufferMemory from Langchain here.
 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def generate(q):
  '''
  Generate output from the correct model and clean it from pre- and post- rambles if possible.
  ''' 
  # make this script retry if the connection is rejected for some reason
  # TODO: use exponential backoff here
  messages.append({"role": "user", "content": q})
  response = openai.ChatCompletion.create(
                model='gpt-4-0314', 
                messages=messages)
  response = response.choices[0].message.content
  # messages.append({"role": "assistant", "content": response})
  
  # clean the response if possible
  c_response = response
  try:
    match = re.search('```coq(.*?)```', c_response, re.DOTALL)
    c_response = match.group(1)
  except:
    pass
  try:
    match = re.search('```(.*?)```', c_response, re.DOTALL)
    c_response = match.group(1)
  except:
    pass
  messages.append({"role": "assistant", "content": response})
  return c_response

def run_trial(q_core, pid, outfile, verbose=True, ntrials=10):
  '''
  Runs one trial on one prompt. 
  - q: function spec with test cases
  - pid: the prompt id
  '''
  q = q_core
  if verbose:
    print("The task: {}".format(q))

  for t in range(ntrials): 
    # for recording the dataset
    out = {
            "prompt_id": pid,
            "iteration": t,
            "instruction": q,
            "output": None,
            "compiler_feedback": None,
            "stats": {
                        "total_lines" : None,
                        "compiled_lines": None,
                        "percent_compiled": None
                    }
            }

    # generate model response
    response = generate(q)

    # get compiler feedback
    try:
      cf = cfeedback(response)
    except:
      print("The weird response: {}".format(response))

    if verbose:
      print("-----Attempt {}---------".format(t))
      print(response)

    if cf is not None:
      line_number = get_linenumber(cf) - 1
      total_lines = get_totallines(response)
      percent_compiled = (line_number)/total_lines
      linetxt = get_line(line_number, response)

      # get the model to reflect on the error
      q = "Your code produces an error in the line {}\n{}Can you please explain what this error means? Let's think step by step. Please rewrite all code if you rewrite any code."\
        .format(linetxt, cf)
      if verbose:
        print(q)
        print(percent_compiled)
    else:
      total_lines = get_totallines(response)
      line_number = total_lines
      percent_compiled = 1.0
      q = "The model solved the problem!"
      if verbose:
        print(q)
        print(percent_compiled)

    # append all data to json lines file
    out["output"] = response
    out["compiler_feedback"] = cf
    out["stats"]["total_lines"] = total_lines
    out["stats"]["compiled_lines"] = line_number
    out["stats"]["percent_compiled"] = percent_compiled

    with open(outfile, 'a') as file:
      file.write(json.dumps(out) + '\n')
    if verbose:
      print("recorded in {}".format(outfile))

    # don't continue if model has completely solved problem
    if cf is None:
      break

  return None

if __name__ == "__main__":
  outfile = "gpt4_coqMBPPTrain01.ndjson"
  # run_trial(q, 0, outfile)
  for i in range(len(dataset)):
    messages=[{"role": "system", "content": systemText}]
    q = dataset[i]['query'] 
    run_trial(q, i, outfile)
