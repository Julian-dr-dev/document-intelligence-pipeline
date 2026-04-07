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
    if not os.path