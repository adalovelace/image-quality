import pandas as pd
import os
import re
import base64
from tqdm import tqdm
from utils.prompts import select_prompt
from utils.cost_calculator import calculate_costs
from utils.gpt_chat import call_gpt4
from adapters.gpt_chat import call_gpt4
from adapters.bedrock_chat import call_aws_bedrock, generate_bedrock_chat, call_aws_bedrock_converse
from utils.models import get_model_id
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"
REGION_NAME = "us-east-1"

category = "clothes"
# category = "sofas"
# category = "handbags"

ADS_PATH = "../datasets/ads.parquet"

PROMPT_TYPE = "criteria" #"criteria" #"generic" #"refined" #

LLM_FEATURES_PATH = f"../datasets/llm_features_{MODEL_NAME}_prompt={PROMPT_TYPE}.parquet"
    
COSTS_FILE = f"../datasets/llm_costs_{MODEL_NAME}_prompt={PROMPT_TYPE}.txt"


def create_client_and_chat(model_id, region_name):
    if "gpt" in model_id:
        client = OpenAI()
        chat = None
    else:
        client, chat = generate_bedrock_chat(model_id, region_name)

    return client, chat

def write_text_to_file(costs, filename):
    with open(filename, "w") as f:
        print(f.write(costs))

def process_responses(df, client, chat, model_id, region_name, prompt, model=None):
    if "nova" in model_id:
        return zip(*[
            call_aws_bedrock_converse(client, model_id, REGION_NAME, prompt, ad_id, img_bytes) 
            for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))
        ])
    elif "gpt" in model_id:
        assert os.environ["OPENAI_API_KEY"] is not None
        return zip(*[
            call_gpt4(client, model_id, prompt, ad_id, img_bytes)
            for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))
        ])
    else:
        return zip(*[
            call_aws_bedrock(chat, model_id, region_name, prompt, ad_id, img_bytes)
            for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))
        ])


if __name__ == "__main__":
    
    df = pd.read_parquet(ADS_PATH)
    print("Dataset")
    print("Shape:", df.shape)
    print(df.head())
    
    max_score = 5
    
    prompt = select_prompt(PROMPT_TYPE, category, max_score)
    
    # # for some reason I can't figure out I get this error
    # # Invalid control character at: line 3 column 232 (char 247)
    # # this is an ugly fix - we should spot the issue in the text instead
    # prompt = remove_invalid_control_characters(prompt)
    
    print("Prompt: ", prompt)
    model_id = get_model_id(MODEL_NAME)
    if not model_id:
        raise ValueError(f"Invalid model: {MODEL_NAME}.")
    
    model, client, chat = None, None, None
    client, chat = create_client_and_chat(model_id, REGION_NAME)

    responses, input_tokens, output_tokens = process_responses(df, client, chat, model_id, 
                                                               REGION_NAME, prompt, model)
        
    
    print(responses[:4])
    
    costs = calculate_costs(MODEL_NAME, sum(input_tokens), sum(output_tokens))
    print("Saving costs to ", COSTS_FILE)
    write_text_to_file(costs, COSTS_FILE)

    llm_df = pd.DataFrame.from_dict(responses) 
    print(llm_df.head())
    llm_df['score'] = llm_df['score'].astype(int)
    
    df_joined = pd.merge(llm_df, df, on="ad_id", how="inner")
    df_joined.to_parquet(LLM_FEATURES_PATH)
    