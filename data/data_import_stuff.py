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
