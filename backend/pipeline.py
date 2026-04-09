import os
import pytesseract
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from typing import List, Dict, Tuple



#config

DPI = 200 
MAX_PAGES = 5  


def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    print(f"  [pipeline] Converting PDF to images: {pdf_path}")

    pages = convert_from_path(
        pdf_path,
        dpi=DPI,
        first_page=1,
        last_page=MAX_PAGES,
    )

    print(f"  [pipeline] {len(pages)} page(s) converted")
    return pages




#extract words + bounding boxes from an image:

def run_ocr(image: Image.Image) -> Dict:

    image = image.convert("RGB")
    width, height = image.size()



    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT
    )

    words = []
    boxes = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()


        if not word or int(data["conf"][i]) < 30:
            continue


        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
 
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h
 
        words.append(word)
        boxes.append([x_min, y_min, x_max, y_max])
 
    print(f"  [pipeline] OCR found {len(words)} words")
 
    return {
        "words":  words,
        "boxes":  boxes,
        "width":  width,
        "height": height,
    }



def normalize_bboxes(
    boxes: List[List[int]],
    width: int,
    height: int
) -> List[List[int]]:
    

    normalized = []

    for box in boxes: 
        x_min, y_min, x_max, y_max = box

        norm_box = [
            int(1000 * x_min / width),
            int(1000 * y_min / height),
            int(1000 * x_max / width),
            int(1000 * y_max / height),
        ]

        norm_box = [max(0, min(1000, v)) for v in norm_box]
        normalized.append(norm_box)

    return normalized

def process_document(pdf_path: str) -> List[Dict]:

    pages = convert_pdf_to_images(pdf_path)
    res = []

    for page_num, page_img, in enumerate(pages, start=1):
        print(f"  [pipeline] Processing page {page_num}/{len(pages)}...")
        

        ocr_result = run_ocr(page_img)

        normalized_boxes = normalize_bboxes(
            ocr_result["boxes"],
            ocr_result["width"],
            ocr_result["height"],
        )

        res.append({
            "page": page_num,
            "image": page_img,
            "words": ocr_result["words"],
            "boxes":  normalized_boxes,
            "width": ocr_result["width"],
            "height": ocr_result["height"],

        })
    print(f"  [pipeline] Document processed — {len(results)} page(s) ready")
    return res

if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_pdf>")
        print("Example: python pipeline.py data/raw/invoice.pdf")
        sys.exit(1)
 
    pdf_path = sys.argv[1]
    results = process_document(pdf_path)
 
    for page in results:
        print(f"\n--- Page {page['page']} ---")
        print(f"Image size: {page['width']} x {page['height']}")
        print(f"Words found: {len(page['words'])}")
        print(f"First 10 words: {page['words'][:10]}")
        print(f"First 3 boxes: {page['boxes'][:3]}")