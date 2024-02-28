import json
import os
import re
from datetime import datetime

RAW_DIR = 'data/gnews_raw'
CLEAN_DIR = 'data/gnews'

if not os.path.exists(CLEAN_DIR):
    os.makedirs(CLEAN_DIR)

for file_name in os.listdir(RAW_DIR):
    file_path = os.path.join(RAW_DIR, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
    date = file_path.split('/')[-1].split('.')[0]
    year, month, day = date.split('-')
    year, month, day = int(year), int(month), int(day)
    clean_data = []
    for item in data:
        format_spec = "%a, %d %b %Y %H:%M:%S %Z"  # "time": "Mon, 01 Jan 2024 08:00:00 GMT"
        parsed_time = datetime.strptime(item['time'], format_spec)
        item_day = parsed_time.day
        if item_day != day:
            continue
        clean_data.append(item)
    clean_file_name = f"{year}-{month:02d}-{day:02d}.json"
    clean_file_path = os.path.join(CLEAN_DIR, clean_file_name)
    with open(clean_file_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
