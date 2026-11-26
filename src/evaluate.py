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
