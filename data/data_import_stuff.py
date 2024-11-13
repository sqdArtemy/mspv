from PIL import Image


def compress_image(image_path, output_path, max_size=(800, 450), quality=85):
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            img.thumbnail(max_size)
            img.save(output_path, "JPEG", quality=quality)
            print(f"Image compressed and saved: {output_path}")
    except Exception as e:
        print(f"Error compressing {image_path}: {e}")
