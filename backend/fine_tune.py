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
        




