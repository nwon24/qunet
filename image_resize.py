from pathlib import Path
from PIL import Image
import argparse


def resize_jpg_dataset(
    input_root,
    output_root,
    target_size=(256, 256)
):
    """
    Resize jpg images and segmentation masks.

    Expected input structure:
        input_root/
        ├── image/
        └── seg/

    Output structure:
        output_root/
        ├── image/
        └── seg/

    Images are resized using bicubic interpolation.
    Segmentation masks are resized using nearest-neighbour interpolation.

    Parameters
    ----------
    input_root : str or Path
        Root directory containing 'image' and 'seg' folders.

    output_root : str or Path
        Root directory where resized 'image' and 'seg' folders will be saved.

    target_size : tuple
        Target size as (width, height).
    """

    input_root = Path(input_root)
    output_root = Path(output_root)

    image_input_dir = input_root / "image"
    seg_input_dir = input_root / "seg"

    image_output_dir = output_root / "image"
    seg_output_dir = output_root / "seg"

    if not image_input_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_input_dir}")

    if not seg_input_dir.exists():
        raise FileNotFoundError(f"Segmentation folder not found: {seg_input_dir}")

    # Create output directories if not present
    image_output_dir.mkdir(parents=True, exist_ok=True)
    seg_output_dir.mkdir(parents=True, exist_ok=True)

    # Resize images
    image_count = 0
    for img_path in image_input_dir.glob("*.jpg"):
        img = Image.open(img_path)

        resized_img = img.resize(
            target_size,
            resample=Image.BICUBIC
        )

        output_path = image_output_dir / img_path.name
        resized_img.save(output_path)

        image_count += 1

    # Resize segmentation masks
    seg_count = 0
    for seg_path in seg_input_dir.glob("*.jpg"):
        mask = Image.open(seg_path)

        resized_mask = mask.resize(
            target_size,
            resample=Image.NEAREST
        )

        output_path = seg_output_dir / seg_path.name
        resized_mask.save(output_path)

        seg_count += 1

    print("Resizing completed.")
    print(f"Images processed: {image_count}")
    print(f"Masks processed: {seg_count}")
    print(f"Resized images saved to: {image_output_dir}")
    print(f"Resized masks saved to: {seg_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Resize jpg images and segmentation masks."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input root directory containing image/ and seg/ folders."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output root directory where resized image/ and seg/ folders will be created."
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Target width. Default is 256."
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Target height. Default is 256."
    )

    args = parser.parse_args()

    resize_jpg_dataset(
        input_root=args.input,
        output_root=args.output,
        target_size=(args.width, args.height)
    )


if __name__ == "__main__":
    main()