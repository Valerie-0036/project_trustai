import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Conv1D
from concrete.ml.torch.hybrid_model import HybridFHEModel

# 1. Load Model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id

# 2. Select ONLY a few layers to encrypt 
# (Encrypting the whole model is too slow for now)
# Let's just try to encrypt the first attention projection of the first block as a test
remote_names = []
for name, module in model.named_modules():
    # Only encrypt the very first attention layer's projection
    if "h.0.attn.c_attn" in name: 
        remote_names.append(name)

print(f"Encrypting layers: {remote_names}")

# 3. Create Hybrid Model
hybrid_model = HybridFHEModel(model, module_names=remote_names)

# 4. FIXED: Define a fixed sequence length
SEQ_LEN = 10 
# We simulate a fixed input size. In FHE, you can't have dynamic sizes.
input_tensor = torch.randint(0, tokenizer.vocab_size, (1, SEQ_LEN), dtype=torch.long)

print("Compiling... (this might take a minute)")
hybrid_model.compile_model(input_tensor, n_bits=8)

# 5. Set to SIMULATION mode first
# "execute" = Real FHE (Wait hours)
# "simulate" = Fast (Checks logic/quantization only)
hybrid_model.set_fhe_mode("simulate") 

prompt = "Programming is a"
inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
inputs = {k: v for k, v in inputs.items()}

print("Generating...")

# 6. Generate with Cache DISABLED
# We must disable cache so the model processes the full SEQ_LEN every time
# otherwise shapes will mismatch the compiled circuit.
with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1, # Generate just 1 token for testing
        use_cache=False,  # CRITICAL: Disable KV cache for FHE compatibility
        pad_token_id=tokenizer.eos_token_id
    )

text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nResult: {text}")