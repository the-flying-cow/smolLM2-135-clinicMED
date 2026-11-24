import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", device_map="auto")

# put in eval mode for testing
model.eval()

print(sum(p.numel() for p in model.parameters()))
print(model.model.layers[0])

prompt= "A 30-year old female presented to the emergency department with"
inputs= tokenizer(prompt, return_tensors= "pt")

with torch.no_grad():
    out_ids= model.generate(
        **inputs,
        max_new_tokens= 60,
        do_sample= True,
        temperature= 0.7,
        top_k= 50,
        pad_token_id= tokenizer.eos_token_id
    )

print(tokenizer.decode(out_ids[0], skip_special_tokens= True))