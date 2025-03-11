
model_name_to_id = {
    "haiku-3":"anthropic.claude-3-haiku-20240307-v1:0",
    "sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "nova-lite": "amazon.nova-lite-v1:0",
    "nova-pro": "amazon.nova-pro-v1:0",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o":  "gpt-4o",
    "gpt-4o-custom": "ft:gpt-4o-2024-08-06:personal:ft-4pochs-bs16:B8OXvjnG",
    "nova-lite-custom": "arn:aws:bedrock:us-east-1:873987660173:provisioned-model/8r21s555tqhy",
    "nova-pro-custom": "arn:aws:bedrock:us-east-1:873987660173:provisioned-model/05uwl1z75viu",
    "Qwen2-5-VL-7B": "Qwen2-5-VL-7B",
    "Qwen2-5-VL-7B-custom-v2": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2-5-VL-72B": "Qwen2-5-VL-72B"
}

def get_valid_models():
    return model_name_to_id.keys()

def get_model_id(model_name):
    return model_name_to_id.get(model_name, None)
