from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

checkpoint = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

model.push_to_hub('my-gpt2-model')
tokenizer.push_to_hub('my-gpt2-model')