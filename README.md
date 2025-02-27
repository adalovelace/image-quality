# Image Quality in Second-Hand Marketplaces
This repository contains the dataset used for the image quality assessment with LLMs as well as the code and prompts used to generate quality scores with LLMs.

## Dataset files

[Ads Metadata](survey_dataset/ads.parquet)

**Fields:** id, title, category, image in bytes and image type as avg/good/bad. The image type is extracted using Claude Sonnet 3.5 with the goal of having images of diverse quality. A few additional images that belonged to screenshots or catalogs were also included.

[User Scores](survey_dataset/user_scores.parquet)

**Fields:** user_id, ad_id, user_score

[User Justifications](survey_dataset/user_justifications.parquet)

Justifications for the scores on a subset of 14 ads from 114 users. Original justifications are in French and have been translated to English using Google Translate.

**Fields:** user_id, ad_id, score, justification in French (answer) and justification in English (answer_eng).

## Notebook

[dataset_exploration.ipynb](dataset_exploration.ipynb)

## Code

[scripts/extract_img_quality_score.py](scripts/extract_img_quality_score.py): Script to extract quality scores from images using different multimodal LLMs.

## Image Quality Aspects Taxonomy

Taxonomy of quality aspects for marketplace item images.

[aspects taxonomy](image_quality_aspects_taxonomy.md)