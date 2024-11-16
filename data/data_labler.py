import os
import pandas as pd
from tkinter import Tk, Label, Button, OptionMenu, StringVar
from tkinter import filedialog
from PIL import Image, ImageTk


activity_options = ["", "archery", "building", "fighting", "mining", "swimming", "walking"]
hearts_options = [""] + [str(i) for i in range(0, 21)]
light_lvl_options = ["", "high", "mid", "low"]
in_hand_item_options = ["", "pickaxe", "sword", "axe", "bow", "crossbow", "block", "miscellaneous"]
target_mob_options = ["", "zombie", "spider", "skeleton", "creeper", "ender", "other"]

decision_mob_options = ["", "go_back", "take_bow", "take_sword"]
decision_light_options = ["", "palce_light_source"]
decision_hearts_options = ["", "give_regeneration_1", "give_regeneration_2", "give_regeneration_3", "give_regeneration_4"]
decision_activity_options = ["", "give_resistance", "give_jump_boost", "give_strength", "give_haste", "give_water_breathing", "give_speed"]


def open_image(image_path: str) -> None:
    image = Image.open(image_path)
    image.thumbnail((500, 500))
    img = ImageTk.PhotoImage(image)
    img_label.config(image=img)
    img_label.image = img


def update_excel() -> None:
    # Get values from dropdowns
    screenshot_title = os.path.basename(image_paths[img_idx])
    activity = activity_var.get()
    hearts = hearts_var.get()
    light_lvl = light_lvl_var.get()
    in_hand_item = in_hand_item_var.get()
    target_mob = target_mob_var.get()
    decision_activity = decision_activity_var.get()
    decision_hearts = decision_hearts_var.get()
    decision_light = decision_light_var.get()
    decision_mob = decision_mob_var.get()

    # Load the Excel file and add a new row
    df = pd.read_excel(excel_file)
    new_row = {
        "screenshot_title": screenshot_title,
        "activity": activity,
        "hearts": hearts,
        "light_lvl": light_lvl,
        "in_hand_item": in_hand_item,
        "target_mob": target_mob,
        "decision_activity": decision_activity,
        "decision_hearts": decision_hearts,
        "decision_light": decision_light,
        "decision_mob": decision_mob
    }
    df = df._append(new_row, ignore_index=True)
    df.to_excel(excel_file, index=False)

    next_image()


def next_image() -> None:
    global img_idx
    img_idx += 1
    if img_idx < len(image_paths):
        open_image(image_paths[img_idx])
    else:
        root.quit()


root = Tk()
root.withdraw()

image_folder = filedialog.askdirectory(title="Select the folder with images")
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

excel_file = filedialog.askopenfilename(title="Select the Excel file", filetypes=[("Excel files", "*.xlsx")])

root.deiconify()


root.title("Minecraft Screenshot Data Entry")
img_label = Label(root)
img_label.pack()


activity_var = StringVar(root)
activity_var.set("")  # Default to blank
Label(root, text="Activity:").pack()
OptionMenu(root, activity_var, *activity_options).pack()

hearts_var = StringVar(root)
hearts_var.set("")
Label(root, text="Hearts:").pack()
OptionMenu(root, hearts_var, *hearts_options).pack()

light_lvl_var = StringVar(root)
light_lvl_var.set("")
Label(root, text="Light Level:").pack()
OptionMenu(root, light_lvl_var, *light_lvl_options).pack()

in_hand_item_var = StringVar(root)
in_hand_item_var.set("")
Label(root, text="In-hand Item:").pack()
OptionMenu(root, in_hand_item_var, *in_hand_item_options).pack()

target_mob_var = StringVar(root)
target_mob_var.set("")
Label(root, text="Target Mob:").pack()
OptionMenu(root, target_mob_var, *target_mob_options).pack()

decision_activity_var = StringVar(root)
decision_activity_var.set("")
Label(root, text="Decision Activity:").pack()
OptionMenu(root, decision_activity_var, *decision_activity_options).pack()

decision_hearts_var = StringVar(root)
decision_hearts_var.set("")
Label(root, text="Decision Hearts:").pack()
OptionMenu(root, decision_hearts_var, *decision_hearts_options).pack()

decision_light_var = StringVar(root)
decision_light_var.set("")
Label(root, text="Decision Light:").pack()
OptionMenu(root, decision_light_var, *decision_light_options).pack()

decision_mob_var = StringVar(root)
decision_mob_var.set("")
Label(root, text="Decision Mob:").pack()
OptionMenu(root, decision_mob_var, *decision_mob_options).pack()


Button(root, text="Next", command=update_excel).pack()


img_idx = 0
if image_paths:
    open_image(image_paths[img_idx])

root.mainloop()
