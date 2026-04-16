
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

def run_inference(model, encoding: Dict) -> torch.Tensor:

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.squeeze(0).argmax(dim=1)

    return predictions



def decode_predictions(processor, encoding: Dict, predictions: torch.Tensor, page: Dict) -> List[Dict]:

    pred_ids = predictions.cpu().tolist()

    word_ids = encoding.word_ids(batch_index=0)

    words  = page["words"]
    boxes  = page["boxes"]

    labeled = []
    seen_word_ids = set()

    for tok_idx, word_idx in enumerate(word_ids):

        
        if word_idx is None:
            continue

        if word_idx is seen_word_ids:
            continue

        seen_word_ids.add(word_idx)

        if word_idx >= len(words):
            continue
        label_id = pred_ids[token_idx]
        label    = ID2LABEL.get(label_id, "O")
 
        labeled.append({
            "word":  words[word_idx],
            "label": label,
            "box":   boxes[word_idx],
        })
 
    return labeled



def extract_fields(labeled_words: List[Dict]) -> Dict[str, str]:
    fields = {
        "vendor":      "",
        "date":        "",
        "invoice_num": "",
        "total":       "",
        "address":     "",
    }
 
    # Map label prefix → field key
    LABEL_TO_FIELD = {
        "VENDOR":      "vendor",
        "DATE":        "date",
        "INVOICE_NUM": "invoice_num",
        "TOTAL":       "total",
        "ADDRESS":     "address",
    }
 
    for item in labeled_words:
        label = item["label"]
        word  = item["word"]
 
        # Skip non-field tokens
        if label == "O":
            continue
 
        # Split "B-DATE" into ("B", "DATE") or "I-VENDOR" into ("I", "VENDOR")
        parts = label.split("-", 1)
        if len(parts) != 2:
            continue
 
        bio_tag, field_name = parts
        field_key = LABEL_TO_FIELD.get(field_name)
 
        if not field_key:
            continue
 
        if bio_tag == "B":
            # B tag — start a new value for this field
            # If the field already has a value, keep only the first occurrence
            if not fields[field_key]:
                fields[field_key] = word
        elif bio_tag == "I":
            # I tag — append this word to the existing field value
            if fields[field_key]:
                fields[field_key] += " " + word
 
    return fields














# 6. FULL EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def extract_from_page(processor, model, page: Dict) -> Dict:
    
    print(f"  [extractor] Encoding page {page['page']}...")
    encoding = encode_page(processor, page)
 
    print(f"  [extractor] Running inference...")
    predictions = run_inference(model, encoding)
 
    print(f"  [extractor] Decoding predictions...")
    labeled_words = decode_predictions(processor, encoding, predictions, page)
 
    print(f"  [extractor] Extracting fields...")
    fields = extract_fields(labeled_words)
 
    return {
        "page":          page["page"],
        "fields":        fields,
        "labeled_words": labeled_words,
    }







#Testing:

if __name__ == "__main__":
    import sys
    from pipeline import process_document
 
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path_to_pdf>")
        sys.exit(1)
 
    pdf_path = sys.argv[1]
 
    # Load model
    processor, model = load_extractor()
 
    # Run pipeline
    pages = process_document(pdf_path)
 
    # Extract from first page
    result = extract_from_page(processor, model, pages[0])
 
    print("\n--- Extracted Fields ---")
    for field, value in result["fields"].items():
        print(f"  {field:15} : {value}")
 
    print("\n--- Labeled Words (first 20) ---")
    for item in result["labeled_words"][:20]:
        print(f"  {item['label']:15} : {item['word']}")
 








