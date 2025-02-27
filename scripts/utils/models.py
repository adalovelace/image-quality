
model_name_to_id = {
    "haiku-3":"anthropic.claude-3-haiku-20240307-v1:0",
    "sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "nova-lite": "amazon.nova-lite-v1:0",
    "nova-pro": "amazon.nova-pro-v1:0",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o":  "gpt-4o",
    "nova-lite-custom": "arn:aws:bedrock:us-east-1:873987660173:provisioned-model/8r21s555tqhy",
    "nova-pro-custom": "arn:aws:bedrock:us-east-1:873987660173:provisioned-model/05uwl1z75viu"
}

def get_model_id(model_name):
    return model_name_to_id.get(model_name, None)