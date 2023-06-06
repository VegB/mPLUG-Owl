"""
CUDA_VISIBLE_DEVICES=? python inference.py \ 
                    --input_csv ../MiniGPT-4/input_csv/visit_bench_single_image.csv \
                    --output_dir ../MiniGPT-4/output_csv/visit_bench_single_image
"""

# Load via Huggingface Style
import os
import urllib.request
from urllib.parse import urlparse
import csv
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer

parser = argparse.ArgumentParser(description="Demo")
parser.add_argument('--input_csv', type=str, default='../MiniGPT-4/input_csv/visit_instructions_700.csv')
parser.add_argument('--output_dir', type=str, default='../MiniGPT-4/output_csv/')
parser.add_argument('--model_name', type=str, default='mPLUG-Owl')
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return csv_reader.fieldnames, data


def add_backslash_to_spaces(url):
    if ' ' in url:
        url = url.replace(' ', "%20")
    return url


def download_image(url, file_path):
    if args.verbose:
        print(url)
        print(file_path)
    try:
        urllib.request.urlretrieve(url, file_path)
        if args.verbose:
            print("Image downloaded successfully!")
    except urllib.error.URLError as e:
        print("Error occurred while downloading the image:", e)


if __name__ == '__main__':
    # check output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_csv = os.path.join(args.output_dir, f'{args.model_name.lower()}.csv')

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
    ).to(device)
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    generate_kwargs = {
        'do_sample': True,
        'top_k': 5,
        'max_length': 512
    }

    # Read CSV file
    fieldname_list, input_data_list = read_csv_file(args.input_csv)

    output_data_list = []
    prediction_fieldname = f'{args.model_name} prediction'
    fieldname_list.append(prediction_fieldname)

    for row in tqdm(input_data_list, total=len(input_data_list), desc='predict'):
        if args.verbose:
            print(row)

        if 'Input.image_url' in row.keys():
            image_url_list = [row['Input.image_url']]
        elif 'image' in row.keys():
            image_url_list = [row['image']]
        else:
            image_url_list = list(eval(row['images']))

        if len(image_url_list) > 1:
            llm_prediction = '[SKIPPED]'

        else:

            # prepare instruction prompt
            prompts = [
            f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <image>
            Human: {row["instruction"]}
            AI: ''']

            # download image image
            image_url = add_backslash_to_spaces(image_url_list[0])
            extension = image_url.split('.')[-1]
            img_path = os.path.join(os.getcwd(), f'tmp.{extension}')  # Create the local image file path
            download_image(image_url, img_path)
            image_list = [img_path]

            images = [Image.open(_) for _ in image_list]
            inputs = processor(text=prompts, images=images, return_tensors='pt')
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                res = model.generate(**inputs, **generate_kwargs)
            llm_prediction = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
            
            if args.verbose:
                print(f'Question:\n\t{row["instruction"]}')
                print(f'Image URL:\t{image_url}')
                print(f'Answer:\n\t{llm_prediction}')
                print('-'*30 + '\n')

            # Clean up
            os.remove(img_path)

        row[prediction_fieldname] = llm_prediction
        output_data_list.append(row)

    # Write to output csv file
    output_file = args.output_csv
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldname_list)
        csv_writer.writeheader()
        csv_writer.writerows(output_data_list)
