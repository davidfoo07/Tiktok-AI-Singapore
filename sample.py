import os
import torch
import yt_dlp
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load AISG_Challenge Dataset
dataset = load_dataset("lmms-lab/AISG_Challenge")

# Function to download video
def download_video(youtube_url):
    video_path = "temp.mp4"
    ydl_opts = {'format': 'best', 'outtmpl': video_path, 'nooverwrites': False}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    return video_path

# Load Model and Processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", device_map=device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Process a test case
def process_test_case(example):
    video_url = example["youtube_url"]
    question = example["question"]
    expected_answer = example["answer"]
    video_path = download_video(video_url)
    
    conversation = [
        {
            "role": "user",
            "content": [
                    {"type": "text", "text": question},
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
    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print("Test Case Output:")
    print(f"Video URL: {video_url}")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Generated Answer: {generated_answer}")

# Run a test case
sample = dataset["test"][0]
process_test_case(sample)