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

    for batch in tqdm(loader, desc="   train", leave=False):

        batch = {k: v.to(cfg.device) for k, v in batch.items()}


        outputs = model(**batch)
        loss    = outputs.loss
 
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
 
        # Clip gradients to prevent exploding gradients
        # This is standard practice when fine-tuning transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
        optimizer.step()
        scheduler.step()
 
        # Compute accuracy for this batch
        predictions = outputs.logits.argmax(dim=-1)
        acc = compute_accuracy(predictions, batch["labels"])
 
        total_loss += loss.item()
        total_acc  += acc
 
    n = len(loader)
    return total_loss / n, total_acc / 0



def val_epoch(model, loader):
    """
    Evaluate the model on the validation set.
    No gradient computation — purely measuring performance.
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
 
    with torch.no_grad():
        for batch in tqdm(loader, desc="  val  ", leave=False):
            batch   = {k: v.to(cfg.device) for k, v in batch.items()}
            outputs = model(**batch)
 
            predictions = outputs.logits.argmax(dim=-1)
            acc = compute_accuracy(predictions, batch["labels"])
 
            total_loss += outputs.loss.item()
            total_acc  += acc
 
    n = len(loader)
    return total_loss / n, total_acc / n














# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*52}")
    print(f"  LayoutLMv2 Fine-tuning on FUNSD")
    print(f"  Device: {cfg.device}")
    print(f"  Epochs: {cfg.epochs}  |  Batch size: {cfg.batch_size}")
    print(f"{'='*52}\n")
 
    # ── Load processor ────────────────────────────────────────────────────────
    print("Loading processor...")
    processor = LayoutLMv2Processor.from_pretrained(
        cfg.model_name,
        revision="no_ocr",
    )
 
    # ── Load datasets ─────────────────────────────────────────────────────────
    train_ds = FUNSDDataset("train", processor)
    val_ds   = FUNSDDataset("test",  processor)
 
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
 
    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading LayoutLMv2 model...")
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
        ignore_mismatched_sizes=True,
    )
    model = model.to(cfg.device)
 
    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    # AdamW is the standard optimizer for fine-tuning transformers
    # It's Adam with weight decay — prevents overfitting
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
 
    # LinearLR gradually warms up the learning rate over training
    # This is standard practice for transformer fine-tuning —
    # starting with a small LR prevents destabilizing pretrained weights
    total_steps = len(train_loader) * cfg.epochs
    scheduler   = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=total_steps // 10,  # warm up over 10% of training
    )
 
    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_loss = float("inf")
 
    print(f"\nStarting training for {cfg.epochs} epochs...\n")
 
    for epoch in range(cfg.epochs):
        print(f"Epoch [{epoch+1}/{cfg.epochs}]")
 
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler
        )
        val_loss, val_acc = val_epoch(model, val_loader)
 
        print(
            f"  train loss={train_loss:.4f}  acc={train_acc:.3f}"
            f"  |  val loss={val_loss:.4f}  acc={val_acc:.3f}"
        )
 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
 
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(cfg.checkpoint_dir)
            processor.save_pretrained(cfg.checkpoint_dir)
            print(f"  [ckpt] Best model saved → {cfg.checkpoint_dir}")
 
        # Periodic checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            epoch_dir = f"{cfg.checkpoint_dir}_epoch{epoch+1}"
            model.save_pretrained(epoch_dir)
            print(f"  [ckpt] Epoch checkpoint → {epoch_dir}")
 
    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {cfg.checkpoint_dir}")
 
    plot_metrics(
        train_losses, val_losses,
        train_accs, val_accs,
        cfg.metrics_path
    )
 
 
if __name__ == "__main__":
    main()
 









