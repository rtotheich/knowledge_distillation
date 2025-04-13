#!/usr/bin/env python 
 
# %% 
MAX_NEW_TOKENS=128 
 
import argparse 
 
parser = argparse.ArgumentParser( 
        prog="llama3.2 evaluation script", 
        description="Evaluates translation capabilities of llama3.2", 
        epilog="Use at your own risk!" 
        ) 
 
parser.add_argument('-m', '--model', help='Model checkpoint to use for evaluation', type=str, required=False) 
 
args = parser.parse_args() 
 
checkpoint = args.model if args.model else "meta-llama/Llama-3.2-1B-Instruct" 
 
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig 
from peft import LoraConfig, get_peft_model, TaskType, PeftModel 
import torch 
import os 
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
set_seed(42) 
 
# %% 
import evaluate 
 
sacrebleu = evaluate.load("sacrebleu") 
comet = evaluate.load("comet") 
 
# %% 
 
model = "meta-llama/Llama-3.2-1B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, 
                                                  device_map="auto", 
                                                  padding_side="left") 
 
print(f"Tokenization is on {tokenizer.padding_side} side") 
 
base_model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto") 
if model != checkpoint: 
    print(f"Evaluating on model {checkpoint}") 
    base_model.resize_token_embeddings(len(tokenizer)) 
    peft_model = PeftModel.from_pretrained(base_model, checkpoint) 
    merged_model = peft_model.merge_and_unload() 
else: 
    print("Evaluating on base model") 
    tokenizer.pad_token = tokenizer.eos_token
    merged_model = base_model 
 
# %% 
merged_model.to(device) 
 
# %% 
source_lang = "en" 
target_lang = "fr" 
 
# %% 
generation_params = { 
    "do_sample": False, 
    "temperature": None, 
    "top_p": None, 
    "eos_token_id": tokenizer.eos_token_id, 
    "pad_token_id": tokenizer.pad_token_id, 
    "max_new_tokens": MAX_NEW_TOKENS, 
    "return_dict_in_generate": True, 
    "output_logits": True, 
} 
 
def inference_chat_template(text): 
    messages = [ 
        {"role": "system", "content": "You are a professional translator. Translate the provided text from English to French, remaining true to the source text. Do not add any additional commentary or conversational elements to your response."}, 
        {"role": "user", "content": text} 
    ] 
    prompt = tokenizer.apply_chat_template( 
        messages, tokenize=False, add_generation_prompt=False 
    ) 
    return prompt 
 
def translate(text): 
    prompt = inference_chat_template(text) 
    inputs = tokenizer(prompt, return_tensors="pt").to(device) 
    with torch.no_grad(): 
        outputs = merged_model.generate(**inputs, **generation_params) 
    prompt_and_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False) 
    response_only = prompt_and_response.split("<|end_header_id|>")
    if len(response_only) > 3:
        response_only = response_only[3].split("�")[0].split("<|begin_of_text|>")[0].split("<|eot_id|>")[0].strip()
    else:
        response_only = ""
    return response_only 
 
def translate_tokenized(inputs): 
    with torch.no_grad(): 
        outputs = merged_model.generate(**inputs, **generation_params) 
    prompt_and_responses = tokenizer.batch_decode([output for output in outputs.sequences], skip_special_tokens=False) 
    responses_only = [output.split("<|end_header_id|>")[3].split("�")[0].split("<|begin_of_text|>")[0].strip() for output in prompt_and_responses] 
    return responses_only 
 
# %% 
from datasets import load_from_disk 
 
dataset_dir = "~/train_llama/left_padded" 
 
test_unlabeled_tokenized = load_from_disk(f"{dataset_dir}/europarl_dataset/test_tokenized_unlabeled") 
test_tokenized = load_from_disk(f"{dataset_dir}/europarl_dataset/test_tokenized") 
 
test_unlabeled_tokenized = test_unlabeled_tokenized.remove_columns(['translation', 'prompt']) 
test_unlabeled_tokenized.set_format("torch", device="cuda") 
 
# %% 
print("Example 0 (French reference) from test dataset:\n") 
print(test_tokenized[0]["translation"]["fr"]) 
print("\n\n") 
 
# %% 
print(translate(test_tokenized[0]["translation"]["en"])) 
 
# %% 
from tqdm.auto import tqdm 
 
sources = [example["translation"]["en"] for example in test_tokenized] 
 
predictions = [] 
 
for example in tqdm(sources): 
    predictions.append(translate(example)) 
 
# %% 
references = [example["translation"]["fr"] for example in test_tokenized] 
 
print(predictions[:5]) 
print(references[:5]) 
 
# %% 
sacrebleu = sacrebleu.compute(predictions=predictions, references=references) 
 
print("SacreBLEU:") 
print(sacrebleu) 
 
# %% 
comet = comet.compute(predictions=predictions, references=references, sources=sources) 
 
print(f"Mean COMET: {comet['mean_score']}") 
 
# %% 
 
 
 
