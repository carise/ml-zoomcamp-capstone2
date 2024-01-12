"""Apply padding to images to prevent distortion when used in the model training.
"""

from PIL import Image, ImageOps
import os

def apply_padding(input_root_dir, output_root_dir, target_size):
	for dirpath, dirnames, filenames in os.walk(input_root_dir):
		print(dirpath, dirnames, len(filenames))

		for image_file in filenames:
			if not image_file.endswith(".jpg"):
				continue
			input_path = os.path.join(dirpath, image_file)
			output_dirpath = dirpath.replace(input_root_dir, output_root_dir)
			output_path = os.path.join(output_dirpath, image_file)

			image = Image.open(input_path)
			image.convert("RGB")

			current_width, current_height = image.size

			pad_width = max(0, target_size[0] - current_width)
			pad_height = max(0, target_size[1] - current_height)

			padded_image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill='black')

			try:
				padded_image.save(output_path)
			except Exception as e:
				print(e)
				print(f"Failed to save image {input_path} to {output_path}")

if __name__ == "__main__":
    input_root_dir = "cleaned_data"
    output_root_dir = "cleaned_padded_data"
    target_size = (300, 300)

    apply_padding(input_root_dir, output_root_dir, target_size)
