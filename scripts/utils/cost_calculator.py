
def calculate_costs(model_name, num_input_tokens, num_output_tokens): 
    # Princing sources: 
    # https://platform.openai.com/docs/pricing
    # https://aws.amazon.com/bedrock/pricing/
    
    if model_name == "nova-pro":
        INPUT_PRICE = 0.0008
        OUTPUT_PRICE = 0.0032
    elif model_name == "sonnet":
        INPUT_PRICE = 0.003
        OUTPUT_PRICE = 0.015
    elif model_name == "nova-lite":
        INPUT_PRICE = 0.00006
        OUTPUT_PRICE = 0.00024
    elif model_name == "haiku-3":
        INPUT_PRICE = 0.00025
        OUTPUT_PRICE = 0.000125
    elif model_name == "haiku-35":
        INPUT_PRICE = 0.001
        OUTPUT_PRICE = 0.0005
    elif model_name == "gpt-4o":
        INPUT_PRICE = 0.0025
        OUTPUT_PRICE = 0.01
    elif model_name == "gpt-4o-mini":
        INPUT_PRICE = 0.00015
        OUTPUT_PRICE = 0.0006
    else:
        return
        
    input_cost = num_input_tokens / 1000 * INPUT_PRICE
    output_cost = num_output_tokens / 1000 * OUTPUT_PRICE
    
    costs_str = f"""
        Input cost (1K tokens) = {INPUT_PRICE}
        Output cost (1K tokens) = {OUTPUT_PRICE}
        -----
        Total input tokens = {num_input_tokens}
        Total output tokens = {num_output_tokens}
        -----
        Total input cost  = {round(input_cost, 6)}$
        Total output cost = {round(output_cost, 6)}$
        Total cost of run = {round(input_cost + output_cost, 6)}$
    """
    print(costs_str)
    
    return costs_str
