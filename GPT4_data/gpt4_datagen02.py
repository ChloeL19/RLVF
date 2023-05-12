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
from langchain.memory import ConversationBufferWindowMemory
import tiktoken as tk

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

openai.organization = "org-f2tK1brD8eM1W91o2X5WgNoy"
openai.api_key = config['OPENAI_KEY']

def cfeedback(v):
  '''
  Returns the compiler error if one exists. Returns None if everything compiles cleanly.
  '''
  r = requests.post("https://coq.livecode.ch/check", data = { 'v': v })
  r.raise_for_status()
  r = r.json()
  
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

def build_dataset(dataset_name):
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
    ds = load_dataset("csv", data_files=dataset_name, split="train")

    def concat(sample):
      sample["query"] = sample['specification'] + "Test case 1: " + sample['test_case_1'] + \
      ", test case 2: " + sample['test_case_2'] + ", test case 3: " + sample['test_case_3']
      return sample

    ds = ds.map(concat, batched=False)
    ds.set_format(type="torch")
    return ds

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset("../MBPP dataset/MBPP_Coq_Train.csv")

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

def add_to_memory(resp):
  # make sure we are uner the token limit.
  # remove earlier model/user interactions, not the system text
  if len(messages) > 5:
    _ = messages.pop(1)
  messages.append(resp)
  
def passes_testcases(resp):
  '''
  TODO: checks that a model's output has met the specification even if it compiles.
  '''
  return True

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(100)) #wait_random_exponential(min=20, max=50)
def generate(q):
    '''
    Generate output from the correct model and clean it from pre- and post- rambles if possible.
    ''' 
    # make this script retry if the connection is rejected for some reason
    add_to_memory({"role": "user", "content": q})
    print("the length of the messages dict: {}".format(len(messages)))
    response = openai.ChatCompletion.create(
                        model='gpt-4', 
                        messages=messages)
    response = response.choices[0].message.content

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
    
    add_to_memory({"role": "assistant", "content": response})
    return c_response

def run_trial(q_core, pid, outfile, verbose=True, ntrials=10):
  '''
  Runs one trial on one prompt. 
  - q_core: function spec with test cases
  - pid: the prompt id
  - outfile: where to save logging results
  '''
  q = q_core
  if verbose:
    print("The task: {}".format(q))

  for t in range(ntrials): 
    passchecks = False
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
    cf = cfeedback(response)

    if verbose:
      print("-----Attempt {}---------".format(t))
      print(response)

    if cf is not None:
      line_number = get_linenumber(cf) - 1
      total_lines = get_totallines(response)
      percent_compiled = (line_number)/total_lines
      linetxt = get_line(line_number + 1, response)

      # get the model to reflect on the error
      q = "Your code produces an error in the line {}: {}\n{}Can you please explain what this error means? Let's think step by step. Please rewrite all code if you rewrite any code."\
        .format(line_number + 1, linetxt, cf)
      if verbose:
        print(q)
        print(percent_compiled)
    else:
      # check for validity of solution, reprompt to actually answer problem.
      if passes_testcases(response):
        passchecks = True
        total_lines = get_totallines(response)
        line_number = total_lines
        percent_compiled = 1.0
        q = "The model solved the problem!"
        if verbose:
          print(q)
          print(percent_compiled)
      else:
        # TODO: fix this part, we need to remprompt the model again to get 
        # back on track
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
    if cf is None and passchecks:
      break

  return None

if __name__ == "__main__":
  outfile = "gpt4_coqMBPPTrain01.ndjson"
  # run_trial(q, 0, outfile)
  for i in range(len(dataset)):
    messages=[{"role": "system", "content": systemText}]
    q = dataset[i]['query'] 
    run_trial(q, i, outfile)
