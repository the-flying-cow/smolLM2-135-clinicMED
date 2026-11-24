import json
import os
import time
import torch
from dataset import ClinicalDomainData
from evaluate import estimate_loss
from lora_model import build_lora_model


def train():

    with open("configs/lora_config.json", "r") as f:
        cfg= json.load(f)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    data_loader= ClinicalDomainData(cfg, device= device)
    model= build_lora_model(cfg, device= device)
    train_params= [p for p in model.parameters() if p.requires_grad]
    optimizer= torch.optim.AdamW(train_params, lr= cfg["learning_rate"])

    start_time= time.time()
    step_time= time.time()

    for step in range(cfg["max_steps"] + 1):

        if step % cfg["eval_interval"] == 0:
        
            metrics= estimate_loss(model, data_loader, eval_iters= cfg["eval_iters"])
            elapsed= time.time() - start_time

            print(
                f"Step {step:3d}/{cfg['max_steps']} | "
                f"Train Loss: {metrics['train']['loss']:.4f} (PPL: {metrics['train']['perplexity']:.2f}) | "
                f"Val Loss: {metrics['val']['loss']:.4f} (PPL: {metrics['val']['perplexity']:.2f}) | "
                f"Time: {elapsed:.1f}s"
            )

        if step == cfg["max_steps"]:
            break

        x, y= data_loader.get_batch("train")
        optimizer.zero_grad(set_to_none= True)
        
        outputs= model(input_ids= x, labels= y)
        loss= outputs.loss
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            dt= time.time() - step_time
            print(f"  Step {step + 1:3d}/{cfg['max_steps']} | Batch Loss: {loss.item():.4f} | 10-step time: {dt:.1f}s")
            step_timer = time.time()

    os.makedirs(cfg["output_dir"], exist_ok= True)
    model.save_pretrained(cfg["output_dir"])

if __name__ == "__main__":
    train()