## Multi_jailbreak_augmented dataset
The dataset originates from the Multilingual Jailbreak Challenges in Large Language Models paper and is publicly available at DAMO-NLP-SG/MultiJail, which contains samples in multiple languages. For this work, a subset of the data was selected, focusing specifically on three languages categorized by resource level:
- English (high-resource)
- Korean (medium-resource)
- Swahili (low-resource)

## 8 prohibited scenarios
The tagging scheme follows the categories defined in the "A Cross-Language Investigation into Jailbreak Attacks in Large Language Models" paper. Instead of using the original dataset tags, the data was reclassified into the following scenario-based categories to better match research goals and improve semantic clarity:
- AdultContent (AC)
- Fraudulent Deceptive Activities (FDA)
- Government Decision Making (GDM)
- HarmfulContent (HC)
- IllegalActivity (IA)
- Political Campaigning Lobbying (PCL)
- UnlawfulPractice (UP)
- ViolatingPrivacy (VP)

## Methodology
1. Classification into these tags was performed using GPT-4o, ensuring consistent semantic labeling aligned with the task of investigating jailbreak attacks.
2. To balance the dataset, augmentation was performed using GPT-4o, which generated at least 30 new samples per tag in a one-shot manner, guided by the provided descriptions.
3. Semantic similarity between original and augmented samples was measured via sentence embeddings, discarding any augmented samples below a 0.75 similarity threshold to maintain data quality and semantic integrity.

## Visit
https://huggingface.co/datasets/YoojongChoi/multi_jailbreak_augmented
