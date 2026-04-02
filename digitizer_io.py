import csv
import json

def save_points_csv(path: str, pts: list[tuple[float, float]], header: tuple[str, str] = ("x", "y")):
    """Saves a list of (x, y) coordinates to a CSV file."""
    with open(path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pt in pts:
            writer.writerow([pt[0], pt[1]])

def save_fit_json(path: str, payload: dict):
    """Saves a Python dictionary to a formatted JSON file."""
    with open(path, mode='w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)