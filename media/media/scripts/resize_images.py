import os
import argparse
from PIL import Image
from tqdm import tqdm
import gc

def resize_and_pad_image(img, target_size, rz_type=Image.Resampling.LANCZOS):
    target_ratio = target_size[0] / target_size[1]
    original_ratio = img.width / img.height

    if original_ratio > target_ratio:
        # The image is wider than the target shape.
        scale_factor = target_size[0] / img.width
    else:
        # The image is taller than the target shape.
        scale_factor = target_size[1] / img.height

    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    img_resized = img.resize(new_size, rz_type)

    # Create a new image with a white background.
    img_padded = Image.new("RGB", target_size, (0, 0, 0))
    # Calculate padding offsets.
    x_offset = (target_size[0] - img_resized.width) // 2
    y_offset = (target_size[1] - img_resized.height) // 2
    # Paste the resized image onto the center of the new image.
    img_padded.paste(img_resized, (x_offset, y_offset))

    return img_padded

def resize_image(input_path, output_path, size, method):
    with Image.open(input_path) as img:
        if method == 'resize':
            img = img.resize(size, Image.Resampling.LANCZOS)
        elif method == 'thumbnail':
            img.thumbnail(size, Image.Resampling.LANCZOS)
        elif method =='pad':
            img = resize_and_pad_image(img, size, Image.Resampling.LANCZOS)
        if output_path is not None: img.save(output_path)
    return img

def walk_and_resize(input_dir, output_dir, size, method):
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in tqdm(files):
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_dir, file)
                resize_image(input_path, output_path, size, method)

    gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Resize images in a directory.')
    parser.add_argument('--input', required=True, help='Input directory path')
    parser.add_argument('--output', required=True, help='Output directory path')
    parser.add_argument('--width', required=True, type=int, help='Width of the image')
    parser.add_argument('--height', required=True, type=int, help='Height of the image')
    parser.add_argument('--method', required=True, choices=['thumbnail', 'resize', 'pad'], help='Resize method')

    args = parser.parse_args()

    size = (args.width, args.height)
    walk_and_resize(args.input, args.output, size, args.method)
