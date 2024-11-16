import pyautogui
import torch

from PIL import Image, ImageTk
from torchvision import transforms
import tkinter as tk
from model.model_import_stuff import num_classes, ds, HeartsInHandClassifier, ImageContextClassifier
from data.data_import_stuff import compress_image, process_screenshot
from interface.constants import icon_paths
from interface.elements import draw_rounded_rect, update_overlay

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

icons = {}
labels = {}
icon_items = {}

for feature, path in icon_paths.items():
    original_icon = Image.open(path)
    resized_icon = original_icon.resize((28, 28), Image.Resampling.LANCZOS)
    icons[feature] = ImageTk.PhotoImage(resized_icon)

for i, feature in enumerate(["activity", "hearts", "light_lvl", "in_hand_item", "target_mob"]):
    y_position = 10 + i * 50
    rounded_rect = draw_rounded_rect(
        10, y_position,
        200, y_position + 40,
        fill="#f0f4f8",
        outline="#c0c8d4",
        width=2, canvas=canvas
    )

    label = tk.Label(canvas, text="", font=("BLOXAT", 10), bg="#f0f4f8", anchor="w", justify="left")
    label.place(x=20, y=y_position + 10)
    labels[feature] = label

    icon_x_position = 175
    icon_y_position = y_position + 20
    icon_item = canvas.create_image(icon_x_position, icon_y_position, image=icons["hearts"], anchor="center")
    icon_items[feature] = icon_item


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

    update_overlay(result=predicted_class, icon_items=icon_items, labels=labels, canvas=canvas, overlay=overlay)
    overlay.after(1000, analyze_gameplay)


overlay.after(1000, analyze_gameplay)
overlay.mainloop()
