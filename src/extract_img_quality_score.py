import pandas as pd
import os
import re
import base64
from tqdm import tqdm
from utils.prompts import select_prompt
from utils.cost_calculator import calculate_costs
from utils.gpt_chat import call_gpt4
from utils.bedrock_chat import call_aws_bedrock, generate_bedrock_chat, call_aws_bedrock_converse
from utils.models import get_model_id
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"
REGION_NAME = "us-east-1"

category = "clothes"
ITEM_TYPE = "clothing item" 

# category = "sofas"
# ITEM_TYPE = "sofa" 

# category = "handbags"
# ITEM_TYPE = "handbag"

ADS_PATH = f"../notebooks/study_dataset/{category}.parquet"

PROMPT_TYPE = "criteria" #"criteria" #"generic" #"refined" #

LLM_FEATURES_PATH = f"../notebooks/study_dataset/llm_features_{category}_{MODEL_NAME}_prompt={PROMPT_TYPE}.parquet"
    
COSTS_FILE = f"../notebooks/study_dataset/llm_costs_{category}_{MODEL_NAME}_prompt={PROMPT_TYPE}.txt"


def create_client_and_chat(model_id, region_name):
    if "gpt" in model_id:
        client = OpenAI()
        chat = None
    else:
        client, chat = generate_bedrock_chat(model_id, region_name)

    return client, chat

def remove_invalid_control_characters(text): 
    pattern = re.compile(r'[^a-zA-Z0-9,. ]')
    # Use the sub() method to replace invalid control characters with an empty string 
    cleaned_text = pattern.sub('', text)
    return cleaned_text


def write_text_to_file(costs, filename):
    with open(filename, "w") as f:
        print(f.write(costs))


if __name__ == "__main__":
    
    df = pd.read_parquet(ADS_PATH)[:3]
    print("Dataset")
    print("Shape:", df.shape)
    print(df.head())
    
    max_score = 5
    screenshot_score = -1
    
    prompt = select_prompt(PROMPT_TYPE, ITEM_TYPE, max_score, screenshot_score)
    
    # # for some reason I can't figure out I get this error
    # # Invalid control character at: line 3 column 232 (char 247)
    # # this is an ugly fix - we should spot the issue in the text instead
    # prompt = remove_invalid_control_characters(prompt)
    
    print("Prompt: ", prompt)
    model_id = get_model_id()
    client, chat = create_client_and_chat(model_id, REGION_NAME)

    if "nova" in MODEL_NAME:
        responses, input_tokens, output_tokens = zip(*[call_aws_bedrock_converse(client, prompt, ad_id, img_bytes)
             for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))])
    elif "gpt" in MODEL_NAME:
            assert os.environ["OPENAI_API_KEY"] is not None
            responses, input_tokens, output_tokens = zip(*[call_gpt4(client, model_id, prompt, ad_id, base64.b64encode(img_bytes).decode("utf-8"))
             for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))])
    else:
        responses, input_tokens, output_tokens = zip(*[call_aws_bedrock(chat, prompt, ad_id, base64.b64encode(img_bytes).decode("utf-8"))
             for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))])
        
    
    print(responses[:4])
    
    costs = calculate_costs(sum(input_tokens), sum(output_tokens))
    print("Saving costs to ", COSTS_FILE)
    write_text_to_file(costs, COSTS_FILE)

    llm_df = pd.DataFrame.from_dict(responses) 
    print(llm_df.head())
    llm_df['score'] = llm_df['score'].astype(int)
    
    df_joined = pd.merge(llm_df, df, on="ad_id", how="inner")
    df_joined.to_parquet(LLM_FEATURES_PATH)
    