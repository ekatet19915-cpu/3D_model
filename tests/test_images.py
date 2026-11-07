import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.pipeline import remove_background, prepare_input

PHOTOS = [
    "source_photos/jpeg/foto-bottle-medium.jpeg",
    "source_photos/jpg/foto-sneaker-medium.jpg",
    "source_photos/jpg/foto-bird-hard.jpg",
]

os.makedirs("outputs", exist_ok=True)
for path in PHOTOS:
    base = os.path.splitext(os.path.basename(path))[0]
    rgba = remove_background(Image.open(path))
    prepared = prepare_input(rgba)
    rgba.save(f"outputs/{base}_nobg.png")
    prepared.save(f"outputs/{base}_input.png")
    print(path, "->", prepared.size)
