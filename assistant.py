import os
import pyautogui
import torch
from PIL import Image
from torchvision import transforms
import tkinter as tk
from model.model_import_stuff import num_classes, ds, HeartsInHandClassifier, ImageContextClassifier
from data.data_import_stuff import compress_image

# Initialize the models
hearts_item_model = HeartsInHandClassifier(hearts_classes=num_classes["hearts"], in_hand_classes=num_classes["in_hand_item"])
hearts_item_model.load_state_dict(torch.load("./model/hearts_model.pth", weights_only=True))
hearts_item_model.eval()

context_model = ImageContextClassifier(num_classes=num_classes)
context_model.load_state_dict(torch.load("./model/context_model.pth", weights_only=True))
context_model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create overlay window for displaying predictions
overlay = tk.Tk()
overlay.overrideredirect(True)
overlay.attributes("-topmost", True)
overlay.wm_attributes("-transparentcolor", "white")
overlay.configure(bg='white')

canvas = tk.Canvas(overlay, bg="white", highlightthickness=0)
canvas.pack(fill="both", expand=True)


def draw_rounded_rect(x1, y1, x2, y2, radius=15, **kwargs):
    points = [
        x1 + radius, y1, x1 + radius, y1,
        x2 - radius, y1, x2 - radius, y1,
        x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2 - radius,
        x2, y2, x2 - radius, y2,
        x1 + radius, y2, x1 + radius, y2,
        x1, y2, x1, y2 - radius,
        x1, y1 + radius, x1, y1 + radius,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


labels = {}
for i, feature in enumerate(["activity", "hearts", "light_lvl", "in_hand_item", "target_mob"]):
    y_position = 10 + i * 70
    rounded_rect = draw_rounded_rect(10, y_position, 260, y_position + 60, fill="#f0f4f8", outline="#c0c8d4", width=2)
    label = tk.Label(canvas, text="", font=("BLOXAT", 12), bg="#f0f4f8", anchor="w", justify="left")
    label.place(x=20, y=y_position + 10)
    labels[feature] = label


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


def update_overlay(result: dict) -> None:
    labels["activity"].config(text=f"Action: {result['activity'].item()}")
    labels["hearts"].config(text=f"Hearts: {result['hearts'].item()}")
    labels["light_lvl"].config(text=f"Light lvl: {result['light_lvl'].item()}")
    labels["in_hand_item"].config(text=f"Item: {result['in_hand_item'].item()}")
    labels["target_mob"].config(text=f"Mob: {result['target_mob'].item()}")

    overlay.update_idletasks()

    width = max(label.winfo_reqwidth() for label in labels.values()) + 40
    height = 10 + len(labels) * 70
    overlay.geometry(f"{width}x{height}+10+10")


def analyze_gameplay() -> None:
    screenshot = pyautogui.screenshot()
    screenshot.save("./static/screenshot.png")

    # Compress image if needed
    compress_image("static/screenshot.png", "static/screenshot.png")
    process_screenshot("./static/screenshot.png", "./static", "./static")

    image = Image.open("static/screenshot.png")
    image = transform(image).unsqueeze(0)

    heart_image = Image.open("static/hearts.png")
    heart_image = transform(heart_image).unsqueeze(0)

    toolbar_image = Image.open("static/toolbar.png")
    toolbar_image = transform(toolbar_image).unsqueeze(0)

    with torch.no_grad():
        hearts_output = hearts_item_model(heart_image, toolbar_image)
        context_out = context_model(image)
        output = {**context_out, **hearts_output}

        features = ["activity", "hearts", "light_lvl", "in_hand_item", "target_mob"]
        predicted_class = {
            feature: ds.label_encoders[feature].inverse_transform(output[feature].argmax(dim=1).cpu().detach().numpy())
            for feature in features
        }

    update_overlay(predicted_class)

    overlay.after(1000, analyze_gameplay)


overlay.after(1000, analyze_gameplay)
overlay.mainloop()
