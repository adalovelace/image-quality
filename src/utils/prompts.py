generic_prompt_func = lambda item_type, max_score: f"""You are an expert at recognising good images for selling items in second-hand marketplaces. Assuming that we can only use one image, and given this image of a {item_type}, provide an overall image quality score for how good is the image if we want to sell it on a second-hand marketplace.
The score should be a number on a scale of 1 to {max_score} (1 and {max_score} included). Do not expect more angles or close up shots as we can only use one image.
If the image has none or minor aspects to improve, please return an overall_score of {max_score}.
Return the score and an justification for the score in JSON format with only the following keys: score, justification. Only return the JSON response with no other explanation."""


criteria_scoring_prompt_func = lambda item_type: f"""You are an expert at recognising good images for selling items in second-hand marketplaces.
Assuming that we can only use one image, and given this image of a {item_type}, provide an overall image quality score for how good is the image if we want to sell it on a second-hand marketplace.
The score should be a number on a scale of 1 to 5 (1 and 5 included). Do not expect more angles or close up shots as we can only use one image.

Follow this criteria when assigning scores:

Score 1: The image is horrible, with many aspects to improve.
Score 2: The image is bad, with a several aspects to improve.
Score 3: The image is not great.
Score 4: The image is quite good, with just 1 or 2 things to improve.
Score 5: The image is fantastic, with none or minor things to improve.

Return the score and an justification for the score in JSON format with only the following keys: score, justification. Only return the JSON response with no other explanation and without including "json" prefix."""


CATEGORY_TO_ITEM_TYPE = {
    "clothes": "clothing item", 
    "sofas": "sofa",
    "handbags": "handbag",
    "all": "second hand item"
}


def select_prompt(prompt_type, category, max_score):
    assert category in CATEGORY_TO_ITEM_TYPE.keys()
    
    item_type = CATEGORY_TO_ITEM_TYPE.get(category, "second hand item")

    if prompt_type == "generic":
        return generic_prompt_func(item_type, max_score)
    elif prompt_type == "criteria":
        return criteria_scoring_prompt_func(item_type)
    else: 
        raise ValueError("Wrong prompt type")
