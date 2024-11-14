import pyautogui
import torch
from PIL import Image
from torchvision import transforms
import tkinter as tk
from model.model_import_stuff import AssistantClassifier, num_classes, ds
from data.data_import_stuff import compress_image

model = AssistantClassifier(num_classes=num_classes)
model.load_state_dict(torch.load("./model/model.pth", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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


def update_overlay(result):
    labels["activity"].config(text=f"Action: {result['activity'].item()}")
    labels["hearts"].config(text=f"Hearts: {result['hearts'].item()}")
    labels["light_lvl"].config(text=f"Light Level: {result['light_lvl'].item()}")
    labels["in_hand_item"].config(text=f"In-hand Item: {result['in_hand_item'].item()}")
    labels["target_mob"].config(text=f"Target Mob: {result['target_mob'].item()}")

    overlay.update_idletasks()

    width = max(label.winfo_reqwidth() for label in labels.values()) + 40
    height = 10 + len(labels) * 70
    overlay.geometry(f"{width}x{height}+10+10")


def analyze_gameplay():
    screenshot = pyautogui.screenshot()
    screenshot.save("./static/screenshot.png")
    compress_image("static/screenshot.png", "static/screenshot.png")

    image = Image.open("static/screenshot.png")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        features = ["activity", "hearts", "light_lvl", "in_hand_item", "target_mob"]
        predicted_class = {
            feature: ds.label_encoders[feature].inverse_transform(output[feature].argmax(dim=1).cpu().detach().numpy())
            for feature in features}

    update_overlay(predicted_class)

    overlay.after(1000, analyze_gameplay)


overlay.after(1000, analyze_gameplay)
overlay.mainloop()
