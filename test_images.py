from PIL import Image
from src.pipeline import remove_background, prepare_input

for name in ["foto-bottle-medium.jpeg", "foto-sneaker-medium.jpg", "foto-bird-hard.jpg"]:
    img = Image.open(name)
    rgba = remove_background(img)
    prepared = prepare_input(rgba)
    base = name.rsplit(".", 1)[0]
    rgba.save(f"outputs/{base}_nobg.png")
    prepared.save(f"outputs/{base}_input.png")
    print(name, "->", prepared.size)