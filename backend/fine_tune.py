import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR


from datasets import load_dataset
from transformers import (
    LayoutLMv2Processor,
    LayoutLMv2ForTokenClassification,
)

from tqdm import tqdm




#config
class Config:
    model_name = "microsoft/layoutlmv2-base-uncased"



    num_labels = 7

    num_labels = 7
    id2label = {
        0: "O",
        1: "B-HEADER",
        2: "I-HEADER",
        3: "B-QUESTION",
        4: "I-QUESTION",
        5: "B-ANSWER",
        6: "I-ANSWER",
    }
    label2id = {v: k for k, v in id2label.items()}

    epochs        = 10
    batch_size    = 2       # keep small — LayoutLMv2 is memory heavy
    lr            = 5e-5    # standard fine-tuning learning rate for transformers
    max_length    = 512     # maximum token sequence length
    save_every    = 2       # save checkpoint every N epochs
 
    # Paths
    checkpoint_dir = "data/checkpoints/layoutlmv2-finetuned"
    metrics_path   = "data/checkpoints/training_metrics.png"
 
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
 
cfg = Config()



#dataset:

class FUNSDDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, processor: LayoutLMv2Processor):

        print(f"  [dataset] Loading FUNSD {split} split:")
        self.dataset = load_dataset("nielsr/funsd")[split]
        self.processor = processor
        print(f"  [dataset] {len(self.dataset)} examples loaded")


    def __len__(self):
        return len(self.dataset)
    


    def __getitem__(self, idx):
        example = self.dataset[idx]

        words = example["words"]
        boxes = example["bboxes"]
        labels = example["ner_tags"]
        image = example["image"].convert("RGB")

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )

        return {k: v.squeeze(0) for k, v in encoding.items()}
    

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:

    mask = labels != -100
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()

    return correct / total if total > 0 else 0.0






#plotting:

def plot_metrics(train_losses, val_losses,
                 train_accs, val_accs, path):
    """Save a 2-panel training metrics chart to disk."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
 
    ax1.plot(train_losses, label="train loss")
    ax1.plot(val_losses,   label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
 
    ax2.plot(train_accs, label="train acc")
    ax2.plot(val_accs,   label="val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Token Accuracy")
    ax2.legend()
 
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [plot] Saved → {path}")





#train an epoch:
def train_epoch(model, loader, omptimizer, scheduler):

    model.train()
    total_loss = 0.0
    total_acc  = 0.0









