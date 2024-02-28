import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
import re
from datetime import datetime
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


RAW_DIR = 'data/gnews_raw'  # 1216 articles in 2024-01
OUTPUT_DIR = 'data/gnews'
NEWS_TIME_FMT = "%a, %d %b %Y %H:%M:%S %Z"
MAX_TOKENS = 512
# DTYPE = torch.bfloat16  # half vram
DTYPE = torch.float16
# DTYPE = torch.float32  # full vram
# MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"  # 18G*3(14G*2); 14(3.5)min/news 
# MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # 16G*2(15G*1); 5.2(0.5)min/news
# MODEL_ID = "gg-hf/gemma-7b-it"  # (20G*1); (2)min/news
MODEL_ID = "gg-hf/gemma-2b-it"  # (10G*1); (0.3)min/news

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generator = pipeline(task='text-generation', model=MODEL_ID, device_map="auto", max_new_tokens=MAX_TOKENS, torch_dtype=DTYPE)


def llm_chat(prompt):
    chat = [
        { "role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    result = generator(prompt)
    response = result[0]['generated_text']
    if response.startswith(prompt):
        response = response[len(prompt):]
    return response


# 5 W's and H: Who? What? When? Where? Why? How?
# news summary including 5 W's and H, sentiment => a structured news
# e.g. Where: America; What: Apple releases VisionPro; Sentiment: positive.
def format_news(item):
    aspects = ["Who", "What", "When", "Where", "Why", "How", "Sentiment"]
    summaries = []
    for aspect in aspects:
        prompt = f'''A financial news could be analyzed using elements including {', '.join(aspects)}. The article is given below:
Title: {item["title"]}.
Content: {item["content"]}.
Briefly summarize the news only in the aspect of {aspect}.'''
        result = llm_chat(prompt)
        summaries.append(result)
    summary = {aspect: summary for aspect, summary in zip(aspects, summaries)}
    return summary


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
news_cnt = 0
for file_name in os.listdir(RAW_DIR):
    file_path = os.path.join(RAW_DIR, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
    date = file_path.split('/')[-1].split('.')[0]
    year, month, day = date.split('-')
    year, month, day = int(year), int(month), int(day)
    output_data = []
    for item in data:
        time = item['time']
        title = item['title']
        content = item['content']
        parsed_time = datetime.strptime(time, NEWS_TIME_FMT)
        item_day = parsed_time.day
        if item_day != day:
            continue

        output_item = format_news(item)
        output_data.append(output_item)
        news_cnt += 1

        break  # DEBUG

    output_file_name = f"{year}-{month:02d}-{day:02d}.json"
    output_file_path = os.path.join(OUTPUT_DIR, output_file_name)
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    break  # DEBUG

print(f"Processed {news_cnt} news.")
