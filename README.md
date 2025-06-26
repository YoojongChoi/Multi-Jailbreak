## 🛡️ MultiJailbreak Fine-Tuning Project
This project focuses on improving the robustness of large language models (LLMs) against jailbreak attacks across multiple languages. The methodology follow the framework presented in the paper "A Cross-Language Investigation into Jailbreak Attacks in Large Language Models".

To enhance safety alignment, LoRA fine-tuning was applied. Notably, Swahili—a low-resource language—showed a dramatic improvement in safety classification accuracy from 29.3% to 95.1%, highlighting the effectiveness of the approach.

Three languages were selected based on resource availability:
- English (high-resource)
- Korean (medium-resource)
- Swahili (low-resource)

---
## 📚 Dataset Construction
The dataset is based on DAMO-NLP-SG/MultiJail. To align with the attack scenarios described in the original paper, several modifications were made:

### 🔖 Semantic Tagging
The original 74 tags from the base dataset were clustered into 8 broader categories:
- AC (Adult Content)
- FDA (Fraudulent Deceptive Activities)
- GDM (Government Decision Making)
- HC (Harmful Content)
- IA (Illegal Activity)
- PCL (Political Campaigning Lobbying)
- UP (Unlawful Practicen)
- VP (Violating Privacy)

A semantic classifier (based on GPT-4o) was used to relabel samples into these 8 tags.

    # After classifying into 8 prohibited scenarios

    AC:   15
    FDA:  13
    GDM:   2
    HC:  165
    IA:  101
    PCL:   0
    UP:    0
    VP:   15
    None:  4
    Total: 315 samples

### 📈 Data Augmentation
To balance the dataset, at least 30 samples per tag were generated using GPT-4o in a one-shot prompting style based on tag-specific descriptions.

    # Augmented Sample Counts:
    # original, augmented
    
    AC:   15 + 15 
    FDA:  13 + 17
    GDM:   2 + 28
    HC:  165 + 0
    IA:  101 + 0
    PCL:   0 + 30
    UP:    0 + 30
    VP:   15 + 15
    None:  4
    
### 🌐 Translation & Filtering
Since the augmented samples were in English only, translations into Korean and Swahili were added using Google Translate. To ensure translation quality, each translated sentence was back-translated to English and compared with the original using sentence embeddings (via sentence-transformers/all-mpnet-base-v2).

#### Filtering Criteria:
Cosine similarity thresholds: 0.70, 0.75, 0.80

Tag-wise data distribution was analyzed for each threshold and 0.75 was chosen as the optimal threshold, balancing quality and quantity

#### Why 0.75?

Higher thresholds (e.g., 0.90) drastically reduced the amount of Swahili data due to translation variability, whereas English and Korean translations remained stable even at higher thresholds. As a result, Swahili's lower similarity scores played a key role in determining the threshold, as a more lenient setting was necessary to preserve sufficient data for analysis.

Sample Counts by Threshold:

Tag	| 0.70	| 0.75 |0.80
--|--|--|--
AC	|28|	25|	23
FDA	|25|	22|	17
GDM	|21|	19|	15
HC	|165|	165|	165
IA	|101|	101|	101
PCL	|23|	18|	12
UP	|17|	16|	8
VP	|30|	30|	28
Total	|410|	396|	369

### 👉 HuggingFace Dataset:
The final filtered dataset using the 0.75 threshold is available here:
https://huggingface.co/datasets/YoojongChoi/multi_jailbreak_augmented

---
## Finetuning 
