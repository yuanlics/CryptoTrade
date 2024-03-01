import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import json
import re
from datetime import datetime
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import time


RAW_DIR = 'data/gnews_raw'  # 1216 articles in 2024-01
OUTPUT_DIR = 'data/gnews'
NEWS_TIME_FMT = "%a, %d %b %Y %H:%M:%S %Z"
MAX_TOKENS = 256
MIN_TOKENS = 128
CONTEXT_LENGTH = 4096
# DTYPE = torch.bfloat16  # half vram
DTYPE = torch.float16
# DTYPE = torch.float32  # full vram
# MODEL_ID = "gpt-3.5-turbo"  # openai api call
# MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"  # fp32: 18G*3, 14min/news; fp16: 14G*2, 3.5min/news
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # fp32: 16G*2, 5.2min/news; fp16: 16G*1, 1min/news
# MODEL_ID = "gg-hf/gemma-7b-it"  # fp16: 20G*1, 2min/news
# MODEL_ID = "gg-hf/gemma-2b-it"  # fp16: 10G*1, 0.3min/news
# MODEL_ID = "facebook/bart-large-cnn"
# ID = 29  # DEBUG

if 'Llama' in MODEL_ID or 'gemma' in MODEL_ID:
    TASK = 'gen'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    generator = pipeline(task='text-generation', model=MODEL_ID, device_map="auto", max_new_tokens=MAX_TOKENS, torch_dtype=DTYPE)
elif 'bart' in MODEL_ID:
    TASK = 'sum'
    summarizer = pipeline("summarization", model=MODEL_ID, device="cuda")


def get_generation(raw_prompt):
    chat = [
        { "role": "user", "content": raw_prompt},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_tokenized = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
    if len(prompt_tokenized) > CONTEXT_LENGTH - MAX_TOKENS:
        print(f"Prompt too long: {len(prompt_tokenized)} tokens. Skip news.")
        return None
    result = generator(prompt)
    response = result[0]['generated_text']
    if response.startswith(prompt):
        response = response[len(prompt):]
    return response


def get_summary(raw_prompt):
    result = summarizer(raw_prompt, max_length=MAX_TOKENS, min_length=MIN_TOKENS, do_sample=False)
    response = result[0]['summary_text']
    return response


# 5 W's and H: Who? What? When? Where? Why? How?
# news summary including 5 W's and H, sentiment => a structured news
# e.g. Where: America; What: Apple releases VisionPro; Sentiment: positive.
def format_news(item):
    if TASK == 'gen':
        prompt = f'Summarize the following financial news. Title: {item["title"]}. Content: {item["content"]}.'
        result = get_generation(prompt)
        if result is None:
            return None
        summary = {'id': item['id'], 'title': item['title'], 'summary': result}
        return summary
    
    if TASK == 'sum':
        prompt = f'Title: {item["title"]}. Content: {item["content"]}.'
        result = get_summary(prompt)
        summary = {'id': item['id'], 'title': item['title'], 'summary': result}
        return summary


def get_raw_file_names(start_ymd, end_ymd):
    start_ymd = args.start_ymd
    end_ymd = args.end_ymd
    start_year, start_month, start_day = map(int, start_ymd.split('-'))
    end_year, end_month, end_day = map(int, end_ymd.split('-'))
    raw_file_names = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for day in range(1, 32):
                if year == start_year and month < start_month:
                    continue
                if year == end_year and month > end_month:
                    continue
                if year == start_year and month == start_month and day < start_day:
                    continue
                if year == end_year and month == end_month and day > end_day:
                    continue
                if month in [4, 6, 9, 11] and day == 31:
                    continue
                if month == 2 and day > 28:  # TODO leap year
                    continue
                raw_file_name = f"{year}-{month}-{day}.json"
                raw_file_names.append(raw_file_name)
    return raw_file_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_ymd", type=str, default='2024-1-21')  # inclusive
    parser.add_argument("--end_ymd", type=str, default='2024-1-30')  # inclusive
    # args = parser.parse_args()
    args = parser.parse_args([])

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    news_cnt = 0
    raw_file_names = get_raw_file_names(args.start_ymd, args.end_ymd)
    tick = time.time()
    for file_name in raw_file_names:
        file_path = os.path.join(RAW_DIR, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)
        date = file_path.split('/')[-1].split('.')[0]
        year, month, day = date.split('-')
        year, month, day = int(year), int(month), int(day)
        output_data = []
        for item in sorted(data, key=lambda x: x['id']):
            ID = item['id']
            time_str = item['time']
            title = item['title']
            content = item['content']
            parsed_time = datetime.strptime(time_str, NEWS_TIME_FMT)
            item_day = parsed_time.day
            if item_day != day:
                continue
            # if ID != 29: continue  # DEBUG

            news_cnt += 1
            tock = time.time()
            print(f"Processing news #{news_cnt}. File: {file_name}. ID: {ID}. Last time: {tock - tick:.2f}s.")
            tick = tock

            output_item = format_news(item)
            if output_item is not None:
                output_data.append(output_item)
            # break  # DEBUG

        output_file_name = f"{year}-{month:02d}-{day:02d}.json"
        output_file_path = os.path.join(OUTPUT_DIR, output_file_name)
        with open(output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        # break  # DEBUG






# BUG: 4k context length too short. llama7b-fp16, 2024-1-22, id 55. 
# SOLUTION: skip; truncate; split
# This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (4096). Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
# Traceback (most recent call last):
#   File "preprocess_news.py", line 77, in <module>
#     output_item = format_news(item)
#   File "preprocess_news.py", line 51, in format_news
#     result = llm_chat(prompt)
#   File "preprocess_news.py", line 33, in llm_chat
#     result = generator(prompt)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/pipelines/text_generation.py", line 241, in __call__
#     return super().__call__(text_inputs, **kwargs)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/pipelines/base.py", line 1196, in __call__
#     return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/pipelines/base.py", line 1203, in run_single
#     model_outputs = self.forward(model_inputs, **forward_params)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/pipelines/base.py", line 1102, in forward
#     model_outputs = self._forward(model_inputs, **forward_params)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/pipelines/text_generation.py", line 328, in _forward
#     generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
#     return func(*args, **kwargs)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/generation/utils.py", line 1592, in generate
#     return self.sample(
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/generation/utils.py", line 2696, in sample
#     outputs = self(
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 1168, in forward
#     outputs = self.model(
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 982, in forward
#     causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)
#   File "/home/liyuan/miniconda3/envs/handy/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 1075, in _update_causal_mask
#     padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
# RuntimeError: The size of tensor a (4096) must match the size of tensor b (4097) at non-singleton dimension 3
