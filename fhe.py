import random
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Conv1D, Trainer, TrainingArguments

from concrete.ml.torch.hybrid_model import HybridFHEModel

# Load the GPT2 model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id

# Determine which layers run with FHE (all linear ones)
remote_names = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, Conv1D)):
        remote_names.append(name)

# Create the HybridFHEModel with the specified remote modules
hybrid_model = HybridFHEModel(model, module_names=remote_names)

# Prepare input data for calibration
input_tensor = torch.randint(0, tokenizer.vocab_size, (1, 32), dtype=torch.long)

# Calibrate and compile the model
hybrid_model.compile_model(input_tensor, n_bits=8, use_dynamic_quantization=True)

hybrid_model.set_fhe_mode("execute")
prompt = "Programming is"
inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
inputs = {k: v for k, v in inputs.items()}

N_TOKENS_GENERATE = 1
# Generate text
with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=N_TOKENS_GENERATE,
        top_p=0.9,
        temperature=0.6,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# Get only the newly generated tokens
input_length = inputs["input_ids"].shape[1]
generated_ids = output[0, input_length:]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# Print the prompt and generated text
print(f"Prompt: {prompt}")
print(f"Response: {generated_text}\n")