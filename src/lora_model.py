import json
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

def build_lora_model(config: dict, device: str= "cpu"):
    base_model= AutoModelForCausalLM.from_pretrained(config["base_model_id"])

    lora_cfg= config["lora"]
    peft_config= LoraConfig(task_type= TaskType.CAUSAL_LM,
                            r= lora_cfg["r"],
                            lora_alpha= lora_cfg["lora_alpha"],
                            lora_dropout= lora_cfg["lora_dropout"],
                            target_modules= lora_cfg["target_modules"],
                            bias= "none")

    model= get_peft_model(base_model, peft_config)
    model.to(device)

    return model

if __name__ == "__main__":
    with open("configs/lora_config.json", "r") as f:
        cfg = json.load(f)

    lora_model = build_lora_model(cfg)

    lora_model.print_trainable_parameters()

    print(lora_model.base_model.model.model.layers[0].self_attn)