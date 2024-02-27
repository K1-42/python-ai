#python -m pip install torch
#python -m pip install transformers
#python -m pip show torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"

# 1.8MB
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('ELYZA-japanese-Llama-2-7b-instruct_tokenizer')

# 9.98GB + 3.50GB
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
model.save_pretrained('ELYZA-japanese-Llama-2-7b-instruct_model')
