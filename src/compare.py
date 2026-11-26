import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_text(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 100) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    torch.manual_seed(7)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_comparison():

    with open("configs/lora_config.json", "r") as f:
        cfg = json.load(f)



    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_id = cfg["base_model_id"]
    adapter_dir = cfg["output_dir"]

    print(f"Loading base model ({base_id}) and LoRA adapter ({adapter_dir}) on {device.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_id)
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    ft_model.eval()


    test_prompts = [
        "A 30-year-old female presented to the emergency department with",
        "Past medical history was significant for",
        "Laboratory investigations revealed a serum creatinine of",
    ]

    for prompt in test_prompts:
        print("\n" + "=" * 70)
        print(f"PROMPT: {prompt}")
        print("=" * 70)

        with ft_model.disable_adapter():
            base_out = generate_text(ft_model, tokenizer, prompt, device)
        print(f"\n[BASE MODEL]:\n{base_out}")

        lora_out = generate_text(ft_model, tokenizer, prompt, device)
        print(f"\n[LORA CLINICAL MODEL]:\n{lora_out}")

if __name__ == "__main__":
    run_comparison()
