from PIL import Image

CUTS = {"styles": (120, 95), "writers": (0, 56)}


def main(folder="paper/figures"):
    for name, (left, top) in CUTS.items():
        picture = Image.open(f"{folder}/{name}.png")
        picture.crop((left, top, *picture.size)).save(f"{folder}/{name}_plain.png")
        print(f"{folder}/{name}_plain.png")


if __name__ == "__main__":
    main()
