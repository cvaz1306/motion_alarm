import cv2
import os
import argparse
import numpy as np


def adjust_brightness(image, adjustment_percent=5):
    """
    Adjust the brightness of every pixel by reducing intensity overall.
    Decreases pixel intensity by a percentage (`adjustment_percent`).

    Args:
        image: Input image (numpy array).
        adjustment_percent: Percent by which to scale down every pixel.

    Returns:
        Processed image with brightness adjusted.
    """
    # Convert adjustment percentage to a scaling factor (reduce by adjustment_percent)
    scale_factor = 1 - (adjustment_percent / 100.0)
    # Apply scaling and clip values to ensure the range is valid (0–255)
    adjusted = np.clip(image * scale_factor, 0, 255).astype(np.uint8)
    return adjusted


def preprocess_images(input_dir, output_dir, adjustment_percent=5):
    """
    Preprocess all images in the input directory and save them to the output directory.

    Args:
        input_dir: Path to the input directory containing images to preprocess.
        output_dir: Path to the output directory where the results will be saved.
        adjustment_percent: The percentage by which pixel intensity will be reduced.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List all files in input directory
    files = os.listdir(input_dir)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    # Process each image
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        try:
            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {input_path}. Skipping...")
                continue

            # Preprocess the image (reduce brightness)
            print(f"Processing: {image_file}")
            preprocessed_image = adjust_brightness(image, adjustment_percent)

            # Save the processed image to output directory
            cv2.imwrite(output_path, preprocessed_image)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess images by reducing pixel intensity by a percentage to avoid naturally black regions.")
    parser.add_argument('--input-dir', type=str, required=True, help="Path to the directory containing input images.")
    parser.add_argument('--output-dir', type=str, required=True, help="Path to the directory where preprocessed images will be saved.")
    parser.add_argument('--adjustment-percent', type=float, default=5, help="Percentage by which to reduce pixel intensity (default=5%).")

    args = parser.parse_args()

    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Adjustment Percent: {args.adjustment_percent}%")

    preprocess_images(args.input_dir, args.output_dir, args.adjustment_percent)


if __name__ == "__main__":
    main()