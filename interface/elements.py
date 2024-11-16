from PIL import Image, ImageTk
from interface.constants import icons_path


def load_icon(activity_type: str) -> ImageTk.PhotoImage:
    icon_path = f"{icons_path}/{activity_type}.png"
    try:
        original_icon = Image.open(icon_path)
        resized_icon = original_icon.resize((32, 32), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized_icon)
    except FileNotFoundError:
        default_icon_path = f"{icons_path}/no_item.png"
        original_icon = Image.open(default_icon_path)
        resized_icon = original_icon.resize((32, 32), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized_icon)


def draw_rounded_rect(x1: int, y1: int, x2: int, y2: int, canvas, radius: int = 15, **kwargs):
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


def update_overlay(result: dict, labels: dict, icon_items: dict, canvas, overlay) -> None:
    labels["activity"].config(text=f"Action: {result['activity'].item()}")
    labels["hearts"].config(text=f"Hearts: {result['hearts'].item()}")
    labels["light_lvl"].config(text=f"Light lvl: {result['light_lvl'].item()}")
    labels["in_hand_item"].config(text=f"Item: {result['in_hand_item'].item()}")
    labels["target_mob"].config(text=f"Mob: {result['target_mob'].item()}")

    activity_type = result["activity"].item()
    activity_icon = load_icon(activity_type)
    canvas.itemconfig(icon_items["activity"], image=activity_icon)

    light_type = result["light_lvl"].item()
    light_icon = load_icon(light_type)
    canvas.itemconfig(icon_items["light_lvl"], image=light_icon)

    item_type = result["in_hand_item"].item()
    item_icon = load_icon(item_type)
    canvas.itemconfig(icon_items["in_hand_item"], image=item_icon)

    mob_type = result["target_mob"].item()
    mob_icon = load_icon(mob_type)
    canvas.itemconfig(icon_items["target_mob"], image=mob_icon)

    icon_items["activity_ref"] = activity_icon
    icon_items["light_ref"] = light_icon
    icon_items["item_ref"] = item_icon
    icon_items["mob_ref"] = mob_icon

    overlay.update_idletasks()

    width = max(label.winfo_reqwidth() for label in labels.values()) + 200
    height = 10 + len(labels) * 70
    overlay.geometry(f"{width}x{height}+10+10")
