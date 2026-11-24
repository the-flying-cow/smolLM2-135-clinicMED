import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer



with open("configs/lora_config.json", "r") as f:
    cfg= json.load(f)



class ClinicalDomainData:
    def __init__(self, config: dict, device: str= "cpu"):
        self.block_size= config["block_size"]
        self.batch_size= config["batch_size"]
        self.device= device

        self.tokenizer= AutoTokenizer.from_pretrained(config["base_model_id"])

        split_str= f"train[:{config['num_samples']}]"
        raw_ds= load_dataset(config["dataset_id"], split= split_str)

        text_col= "full_note" if "full_note" in raw_ds.column_names else "note"
        all_ids= []

        eos_id= self.tokenizer.eos_token_id

        for row in raw_ds:
            text= (row[text_col] or "").strip()
            if text:
                ids= self.tokenizer.encode(text, add_special_tokens= False)
                all_ids.extend(ids)
                if eos_id is not None:
                    all_ids.append(eos_id)

        tokens= torch.tensor(all_ids, dtype= torch.long)
        n= int(config["train_split"] * len(tokens))

        self.train_data= tokens[: n]
        self.val_data= tokens[n: ]

        print(
            f"Total tokens: {len(tokens):,} | "
            f"Train: {len(self.train_data):,} | Val: {len(self.val_data):,}"
        )

    def get_batch(self, split: str):
        data= self.train_data if split == "train" else self.val_data

        idx= torch.randint(len(data) - self.block_size, (self.batch_size, ))
        x= torch.stack([data[i: i + self.block_size] for i in idx]).to(self.device)
        return x, x.clone()

loader= ClinicalDomainData(cfg)
x, y= loader.get_batch("train")

print(loader.tokenizer.decode(x[0]))