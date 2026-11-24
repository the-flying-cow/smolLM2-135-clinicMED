import json
import math
import torch
from dataset import ClinicalDomainData
from lora_model import build_lora_model

@torch.no_grad()

def estimate_loss(model, data_loader: ClinicalDomainData, eval_iters: int) -> dict:
    model.eval()
    results= {}

    for split in ["train", "val"]:
        losses= torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y= data_loader.get_batch(split)
            output= model(input_ids= x, labels= y)
            losses[k]= output.loss.item()

        mean_loss= losses.mean().item()
        results[split]= {
            "loss": mean_loss,
            "perplexity": math.exp(mean_loss)
        }

    model.train()
    return results

if __name__ == "__main__":
    with open("configs/lora_config.json", "r") as f:
        cfg= json.load(f)

    data_loader= ClinicalDomainData(cfg, device= "cpu")
    model= build_lora_model(cfg, device= "cpu")

    print(f"Step - 0 unadapted weights")
    metrics= estimate_loss(model, data_loader, eval_iters= cfg["eval_iters"])

    print(
        f"Train Loss: {metrics['train']['loss']:.4f} |"
        f"Train PPL: {metrics['train']['perplexity']:.2f}"
    )
    print(
        f"Val Loss:   {metrics['val']['loss']:.4f} |"
        f"Val PPL:   {metrics['val']['perplexity']:.2f}"
    )