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
