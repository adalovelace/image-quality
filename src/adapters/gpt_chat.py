
import json
import re
import base64

"""
Sometimes the response returns some text before and/or after the JSON file.
This method cleans the response and just returns the JSON part.
"""
def clean_content_and_return_json(text):
    text = text.replace("\n", "").replace("{ ","{").replace("{  ","{")
    
    json_pattern = r'({.*?})'
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(1)
        
        try:
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError:
            print("Invalid JSON object found.")
            return None
    else:
        print("No JSON object found in the string.")
        return None
    
    
def call_gpt4(client, model_id, prompt, ad_id, img_bytes):
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    result_quality = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at recognising good images for selling items in second-hand marketplaces."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            }
        ],
        max_tokens=500
    )
    
    response_text = result_quality.choices[0].message.content
    input_num_tokens = result_quality.usage.prompt_tokens
    output_num_tokens = result_quality.usage.completion_tokens
    
    response_json = clean_content_and_return_json(response_text)
    print(response_json)
    
    response_json["ad_id"] = ad_id
    
    return response_json, input_num_tokens, output_num_tokens
