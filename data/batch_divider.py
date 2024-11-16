import os
import shutil


def divide_screenshots(source_dir: str, destination_dir: str) -> None:
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    files.sort()

    batch_size = len(files) // 10
    if len(files) % 10 != 0:
        batch_size += 1

    for i in range(10):
        batch_folder = os.path.join(destination_dir, f"batch_{i + 1}")
        os.makedirs(batch_folder, exist_ok=True)

        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        for file in files[start_idx:end_idx]:
            shutil.move(os.path.join(source_dir, file), os.path.join(batch_folder, file))

    print("Screenshots divided into 10 batches successfully.")


source_directory = ".//raw_images"
destination_directory = ".//raw_images"
divide_screenshots(source_directory, destination_directory)
