# Image Quality in Second-Hand Marketplaces
This repository contains the dataset used for the image quality assessment with LLMs as well as the code and prompts used to generate quality scores with LLMs.

## Dataset files

[Ads Metadata](datasets/ads.parquet)

**Fields:** id, title, category, image in bytes and image type as avg/good/bad. The image type is extracted using Claude Sonnet 3.5 with the goal of having images of diverse quality. A few additional images that belonged to screenshots or catalogs were also included.

[User Scores](datasets/user_scores.parquet)

**Fields:** user_id, ad_id, user_score

[User Justifications](datasets/user_justifications.parquet)

Justifications for the scores on a subset of 14 ads from 114 users. Original justifications are in French and have been translated to English using Google Translate.

**Fields:** user_id, ad_id, score, justification in French (answer) and justification in English (answer_eng).

## Image Quality Aspects Taxonomy

Taxonomy of quality aspects for marketplace item images.

[aspects taxonomy](image_quality_aspects_taxonomy.md)

## Notebooks

[dataset_exploration.ipynb](notebooks/00_dataset_exploration.ipynb)

## Code

[src/extract_img_quality_score.py](scripts/extract_img_quality_score.py): Script to extract quality scores from images using different multimodal LLMs. 

You can run it with the following command.
```
python extract_img_quality_score.py --model <model_name> --prompt <prompt_type> --region <region_name>
```
By default, the AWS region is `us-east-1`.

Example:
```
python extract_img_quality_score.py --model gpt-4o-mini --prompt generic
```

You will need to setup LLM permissions to be able to call the supported LLMs (see below).

### Supported models
* Anthropic's Claude Haiku 3
* Anthropic's Claude Sonnet 3.5
* Amazon Nova Lite 
* Amazon Nova Pro
* OpenAI GPT 4o mini
* OpenAI GPT 4o
* Qwen2-5-VL-7B
* Qwen2-5-VL-72B

The list of models can also be found in [src/utils/models.py].

### Setting up LLM permissions 

#### Bedrock permissions
Currently the notebooks and scripts using Bedrock LLMs only run in the `p10n-vs-dev-01` Unicron cluster using this role `arn:aws:iam::873987660173:role/unicron-bedrock-and-s3` which allows to call AWS Bedrock. This role is also set up to have  access to Databox by adding a trust relationship with our Databox role (note that this has to be done both on our AWS account as well as from DataHub Permissions).

To enable them in other environments we should add this role (or create a new one) to the Unicron cluster by asking CRE in #unicron-users channel.

To use Bedrock, we need to manually enable the models we want to use in the AWS Console. Currently only **Text & Vision** models have been enabled.

![enable_models](enable_models.png)

#### GPT permissions

To use GPT-4o and GPT-4o-mini you just need an API KEY which you can get from (https://platform.openai.com/api-keys)[https://platform.openai.com/api-keys].

#### Setting up inference with Qwen2-VL

To run inference on Qwen2-VL you will need to setup an inference machine, either locally or in the cloud (e.g. EC2 instance). For the experiments we used 2 types of [G5 instances](https://aws.amazon.com/es/ec2/instance-types/g5/): a g5x2 for 7B and a g5x48 for 72B. 

To install Qwen2-VL:
```
pip install git+https://github.com/huggingface/transformers accelerate
```

For more details see [https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct/summary](https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct/summary).