import base64
import json
import time
import boto3
from botocore.config import Config
from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage


def generate_bedrock_chat(model_id, region_name):
    # Setting up an AWS bedrock client
    bedrock_config = Config(
        region_name=region_name,
        retries={
            "max_attempts": 5,
            "mode": "standard",
        },
        # read_timeout=500, # When you are dealing with a large prompt you need to increase the read_timeout
    )
    
    session = boto3.Session()
    credentials = session.get_credentials().get_frozen_credentials()

    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        aws_session_token=credentials.token,
        config=bedrock_config,
    )
    
    chat = ChatBedrock(
        client=bedrock_client,
        model_id=model_id,
        provider="anthropic",
        model_kwargs={"temperature": 0, "max_tokens": 300}
    )
    
    return bedrock_client, chat


def call_aws_bedrock_converse(client, model_id, region_name, prompt, ad_id, img_bytes):
    retry_count = 0
    response_as_dict = {}
    
    while retry_count < 5:
        try:
            system = [{ "text": "You are an expert at image quality" }]

            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "image": { 
                                "format": "jpeg",
                                "source": {
                                    "bytes": img_bytes 
                                }
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                 },
            ]
            
            inf_params = {"maxTokens": 1000, "topP": 0.1, "temperature": 0}
            additionalModelRequestFields = {
                "inferenceConfig": {
                    "topK": 20
                }
            }
            response = client.converse(
                modelId=model_id, 
                messages=messages, 
                system=system, 
                inferenceConfig=inf_params,
                additionalModelRequestFields=additionalModelRequestFields
            )
            print("\n[Full Response]")
            print(json.dumps(response["output"]["message"]["content"][0]["text"], indent=2))
            
            response_text = response["output"]["message"]["content"][0]["text"]
            num_input_tokens = response["usage"]["inputTokens"]
            num_output_tokens = response["usage"]["outputTokens"]
            
            if response_text[-1] != '}':
                response_text = response_text + '\"}'
            response_as_dict = json.loads(response_text)
            response_as_dict["ad_id"] = ad_id
            return response_as_dict, num_input_tokens, num_output_tokens
        
        except Exception as e:
            print("Exception: ", e)
            retry_count += 1
            wait_time = 2 ** retry_count  # Exponential backoff
            print("Recreating chat...")
            client, chat = generate_bedrock_chat(model_id, region_name)
            print("Waiting....")
            time.sleep(wait_time)
    
    # If all retries fail, return default values
    response_as_dict["score"] = -1
    response_as_dict["justification"] = ""
    response_as_dict["ad_id"] = ad_id
    return response_as_dict, 0, 0


def call_aws_bedrock(chat, model_id, region_name, prompt, ad_id, img_bytes):
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    retry_count = 0

    while retry_count < 5:
        try:
            objects_multimodal_messages = [
                SystemMessage(content="You are an expert at image quality"),
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                )
            ]
            response = chat.invoke(objects_multimodal_messages)

            print("Response:", response.content)
            
            response_as_dict = json.loads(response.content)
            response_as_dict["ad_id"] = ad_id

            # Extract num tokens for cost estimate
            num_input_tokens = response.usage_metadata["input_tokens"]
            num_output_tokens = response.usage_metadata["output_tokens"]

            return response_as_dict, num_input_tokens, num_output_tokens
        
        except Exception as e:
            print("Exception: ", e)
            retry_count += 1
            wait_time = 2 ** retry_count  # Exponential backoff
            print("Recreating chat...")
            client, chat = generate_bedrock_chat(model_id, region_name)
            print("Waiting....")
            time.sleep(wait_time)
    
    # If all retries fail, return default values
    return {}, 0, 0