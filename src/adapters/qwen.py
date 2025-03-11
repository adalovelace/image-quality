import re
import torch
import base64
import json
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from PIL import Image
import io
from openai import OpenAI


def create_qwen2_model(model_id, model_adapter=None):
    if "Qwen2-5-VL-7B" in model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto")
    elif "Qwen2-5-VL-7B-custom" in model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto")
        print("Loading adapter from HF")
        model.load_adapter(model_adapter)
    else:
        raise ValueError(f"Qwen2 Model with id {model_id} not supported")
    

def remove_json_prefix(response_text: str) -> str:
    """
    Remove leading '```json' and trailing '```' from the response text.
    """
    cleaned = response_text.strip()
    # Remove leading ```json and trailing ```
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned


def call_local_model_transformers(model, model_id, prompt, ad_id, img_bytes): 
    image = Image.open(io.BytesIO(img_bytes))
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    messages = [
        {  
            "role": "system",
            "content": "You are an expert at image quality"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},

                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preparation for inference
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print("Response:", output_text[0])
    
    response_as_dict = json.loads(remove_json_prefix(output_text[0]))
    response_as_dict["ad_id"] = ad_id
    return response_as_dict, 0, 0


def call_local_model_openai_server(client, model_id, prompt, ad_id, img_bytes):
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {  
            "role": "system", "content": "You are an expert at image quality"
            },
            {
                "role": "user",
                "content": [
                    
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    output_text = response.choices[0].message.content
    print(output_text)
    response_as_dict = json.loads(remove_json_prefix(output_text))
    response_as_dict["ad_id"] = ad_id
    return response_as_dict, 0, 0
