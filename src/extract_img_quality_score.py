import pandas as pd
import os
import argparse
from tqdm import tqdm
from utils.prompts import select_prompt
from utils.cost_calculator import calculate_costs
from utils.gpt_chat import call_gpt4
from adapters.gpt_chat import call_gpt4
from adapters.bedrock_chat import call_aws_bedrock, generate_bedrock_chat, call_aws_bedrock_converse
from adapters.qwen import create_qwen2_model, call_local_model_transformers, call_local_model_openai_server
from utils.models import get_model_id, get_valid_models
from openai import OpenAI

DEFAULT_MODEL_NAME = "sonnet"
DEFAULT_REGION_NAME = "us-east-1"
QWEN_MODEL_ADAPTER = "user/qwen2-5-VL-7b-instruct-trl-sft-iq-v1"

DEFAULT_PROMPT_TYPE = "criteria"
DEFAULT_CATEGORY = "all"

ADS_PATH = "../datasets/ads.parquet"


def resolve_output_path(model_name, prompt_type):
    return f"../datasets/llm_features_{model_name}_prompt={prompt_type}.parquet"


def resolve_costs_path(model_name, prompt_type):
    return f"../datasets/llm_costs_{model_name}_prompt={prompt_type}.txt"


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
            call_aws_bedrock_converse(client, model_id, DEFAULT_REGION_NAME, prompt, ad_id, img_bytes) 
            for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))
        ])
    elif "gpt" in model_id:
        assert os.environ["OPENAI_API_KEY"] is not None
        return zip(*[
            call_gpt4(client, model_id, prompt, ad_id, img_bytes)
            for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))
        ])
    elif "Qwen2-5-VL-7B" in model_id:
        assert model is not None
        return zip(*[
            call_local_model_transformers(model, model_id, prompt, ad_id, img_bytes)
            for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))
        ])
    elif "Qwen2-5-VL-72B" in model_id:
        return zip(*[
            call_local_model_openai_server(client, model_id, prompt, ad_id, img_bytes)
            for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))
        ])
    else:
        return zip(*[
            call_aws_bedrock(chat, model_id, region_name, prompt, ad_id, img_bytes)
            for ad_id, img_bytes in tqdm(zip(df['ad_id'], df['image']))
        ])


def main(model_name, prompt_type, region_name=DEFAULT_REGION_NAME):
    
    df = pd.read_parquet(ADS_PATH)
    print("Dataset")
    print("Shape:", df.shape)
    print(df.head())
    
    max_score = 5
    category = DEFAULT_CATEGORY
    prompt = select_prompt(prompt_type, category, max_score)
    
    print("Prompt: ", prompt)
    model_id = get_model_id(model_name)
    if not model_id:
        raise ValueError(f"Invalid model: {model_name}.")
    
    model, client, chat = None, None, None
    client, chat = create_client_and_chat(model_id, region_name)

    responses, input_tokens, output_tokens = process_responses(df, client, chat, model_id, 
                                                               region_name, prompt, model)
        
    
    print(responses[:4])
    
    costs = calculate_costs(model_name, sum(input_tokens), sum(output_tokens))
    llm_costs_path = resolve_costs_path(model_name, prompt_type)
    print("Saving costs to ", llm_costs_path)
    write_text_to_file(costs, llm_costs_path)

    llm_df = pd.DataFrame.from_dict(responses) 
    print(llm_df.head())
    llm_df['score'] = llm_df['score'].astype(int)
    
    llm_scores_path = resolve_output_path(model_name, prompt_type)
    df_joined = pd.merge(llm_df, df, on="ad_id", how="inner")
    df_joined.to_parquet(llm_scores_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score extraction parameters.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, choices=get_valid_models(), help="Name of the model to use.")
    parser.add_argument("--region", type=str, default=DEFAULT_REGION_NAME, help="Region name for the model.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT_TYPE, choices=["generic", "criteria"], help="Type of prompt to use.")
    
    args = parser.parse_args()

    main(model_name=args.model, region_name=args.region, prompt_type=args.prompt)
