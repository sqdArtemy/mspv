import os

from PIL import Image


def compress_image(image_path: str, output_path: str, max_size: tuple[int, int] = (800, 450), quality: int = 85) -> None:
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            img.thumbnail(max_size)
            img.save(output_path, "JPEG", quality=quality)
    except Exception as e:
        print(f"Error compressing {image_path}: {e}")


def process_screenshot(screenshot_path: str, hearts_output_folder: str, in_hand_output_folder: str) -> None:
    os.makedirs(hearts_output_folder, exist_ok=True)
    os.makedirs(in_hand_output_folder, exist_ok=True)

    heart_region_coords = (215, 370, 400, 420)
    in_hand_region_coords = (200, 410, 600, 450)

    image = Image.open(screenshot_path).convert('RGB')

    heart_region = image.crop(heart_region_coords)
    heart_filename = os.path.join(hearts_output_folder, "hearts.png")
    heart_region.save(heart_filename)

    in_hand_region = image.crop(in_hand_region_coords)
    in_hand_filename = os.path.join(in_hand_output_folder, "toolbar.png")
    in_hand_region.save(in_hand_filename)
