
import torch
from PIL import Image
from transformers import (
    LayoutLMv2Processor,
    LayoutLMv2ForTokenClassification,
)
from typing import List, Dict

LABELS = [
    "O",             
    "B-VENDOR",     
    "I-VENDOR",      
    "B-DATE",        
    "I-DATE",        
    "B-INVOICE_NUM", 
    "I-INVOICE_NUM", 
    "B-TOTAL",       
    "I-TOTAL",       
    "B-ADDRESS",     
    "I-ADDRESS",     
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}



NUM_LABELS = len(LABELS)


#device config: 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_extractor():
    print("[extractor] Loading LayoutLMv2 processor...")
    processor = LayoutLMv2Processor.from_pretrained(
        "microsoft/layoutlmv2-base-uncased",
        revision="no_ocr",
    )

    print("[extractor] Loading LayoutLMv2 model...")
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv2-base-uncased",
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
 
    model = model.to(DEVICE)
    model.eval()
 
    print(f"[extractor] Model ready on {DEVICE}")
    return processor, model

def encode_page(processor, page:Dict) -> Dict:


    words = page["words"]
    boxes = page["boxes"]
    image = page["images"]


    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    return encoding

#def run_inference(model, encoding: Dict) -> torch.Tensor:
