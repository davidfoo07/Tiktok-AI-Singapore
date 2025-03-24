import os
import torch
import yt_dlp
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import time

start_time = time.time()

# Load AISG_Challenge Dataset
dataset = load_dataset("lmms-lab/AISG_Challenge")

# Load Model and Processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", 
    torch_dtype=torch.float16,
    device_map=device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Function to download video
def download_video(youtube_url):
    filename = f"{youtube_url.split("/")[-1]}.mp4"
    
    if (not os.path.exists("./videos")):
        os.mkdir("./videos")
        
    video_path = f"./videos/{filename}"
    
    if (not os.path.exists(video_path)):  
        ydl_opts = {
            'format': 'best', 
            'outtmpl': video_path, 
            'nooverwrites': False
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url)
            title = info.get("title", "Unknown Title")
    else:
        ydl_opts = {
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            title = info.get("title", "Unknown Title")
    
    return video_path, title

# Process a test case
def process_test_case(example):
    video_url = example["youtube_url"]
    question = example["question"]
    question_prompt = example["question_prompt"]
    expected_answer = example["answer"]
    video_path, title = download_video(video_url)
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Summarize the video in details."},
                {"type": "video", "path": video_path},
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(
            conversation,
            video_fps=1,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    video_summary = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    conversation = [
        {
            "role": "user",
            "content": [
                    {"type": "text", "text": f"Video title: {title}\nVideo summary: {video_summary}\nQuestion:\n{question}\n{question_prompt}"},
                    {"type": "video", "path": video_path},
                ],
        },
    ]
    
    inputs = processor.apply_chat_template(
            conversation,
            video_fps=1,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    print(f"Video URL: {video_url}")
    print(f"Question:\n{question}\n{question_prompt}")
    print(f"Answer: {generated_answer}")

# Run a test case
sample = dataset['test'].filter(lambda x: x['qid'] == "0008-0")[0]
process_test_case(sample)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")