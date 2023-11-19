import base64
import requests
import io
from PIL import Image
import os
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import argparse


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class LogoPredictor:
    def __init__(self, key):
        self.key = key

    def predict_openai(self, image_path):
        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Can you identify the logo present in the image? give the name of the company separated by commas."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()

    def predict_google(self, image_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.key
        client = vision_v1.ImageAnnotatorClient()

        pil_image = Image.open(image_path)
        img_byte_array = io.BytesIO()
        pil_image.save(img_byte_array, format="JPEG")
        image_content = img_byte_array.getvalue()

        response = client.logo_detection(image={"content": image_content})
        logos = response.logo_annotations

        logo_tags_dict = {}
        for index, logo in enumerate(logos):
            if logo.score > 0.5:
                logo_tags_dict[index] = {'logo': logo.description}

        return logo_tags_dict
    
    def find_unique_logo(self, logo_dict):
        unique = []
        for k, v in logo_dict.items():
            if v['logo'] not in unique:
                unique.append(v['logo'])
        return ', '.join(unique)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict logos using OpenAI or Google Cloud Vision')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('predictor', type=str, choices=['openai', 'google'], help='Choose the predictor (openai or google)')
    
    args = parser.parse_args()

    if args.predictor == 'openai':
        #add openai gpt4 vision api key here
        logo_predictor = LogoPredictor(key="add-api-key-here")
        result = logo_predictor.predict_openai(args.image_path)
        content = result['choices'][0]['message']['content']
    elif args.predictor == 'google':
        #add google service account credentials file path here
        logo_predictor = LogoPredictor(key='add-service-path-here')
        result = logo_predictor.predict_google(args.image_path)
        content = logo_predictor.find_unique_logo(result)

    print(content)
