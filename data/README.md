# UniFER Datasets

This folder contains the dataset files used for training and evaluating the UniFER model.

## Dataset Files

### Training Datasets

1. **UniFER_CoT_230K.json** - Chain-of-Thought dataset for cold-start supervised fine-tuning
   - Contains 230K samples with facial expression recognition tasks
   - Each sample includes:
     - `image_path`: Path to the image file
     - `question`: The facial expression recognition question
     - `response`: The answer in CoT format with `<think>` and `<answer>` tags

2. **UniFER_RLVR_360K.json** - Reinforcement Learning with Verifiable Rewards dataset
   - Contains 360K samples for RLVR training
   - Same structure as UniFER_CoT_230K.json

### Evaluation Datasets

The following datasets are used for benchmarking and evaluation:

1. **rafdb_qa.json** - RAF-DB dataset in question-answering format
   - Facial Expression Recognition Database
   - 7 emotion categories: surprise, fear, disgust, happiness, sadness, anger, neutral

2. **ferplus_qa.json** - FERPlus dataset in question-answering format
   - Extended Facial Expression Recognition dataset
   - 8 emotion categories: angry, contempt, disgust, fear, happy, neutral, sad, surprise

3. **affectnet_qa.json** - AffectNet dataset in question-answering format
   - Large-scale facial expression database
   - 8 emotion categories: neutral, happiness, sadness, surprise, fear, disgust, anger, contempt

4. **sfew_2.0_qa.json** - SFEW 2.0 dataset in question-answering format
   - Static Facial Expressions in the Wild dataset
   - 7 emotion categories: surprise, fear, disgust, happiness, sadness, anger, neutral

## Data Format

### Training Data Format
```json
[
  {
    "image_path": "/RAF-DB/aligned/train_00001_aligned.jpg",
    "question": "As an expert in facial expression recognition, which expression is most prominent in this image? Please select your answer from the following candidate labels: ...",
    "response": "<think>Reasoning process...</think><answer>emotion_label</answer>"
  }
]
```

### Evaluation Data Format
```json
[
  {
    "image_path": "/RAF-DB/aligned/test_3068_aligned.jpg",
    "true_label": "neutral",
    "prompt": "As an expert in facial expression recognition, which expression is most prominent in this image? ...",
    "candidate_labels": ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
  }
]
```

## Usage

### For Training

The training scripts expect the following directory structure:
```
UniFER/
├── data/
│   ├── UniFER_CoT_230K.json
│   └── UniFER_RLVR_360K.json
└── dataset/  (contains actual image files)
    ├── RAF-DB/
    ├── FERPlus/
    ├── AffectNet/
    └── SFEW2.0/
```

**Note**: The actual image files are NOT included in this repository and must be downloaded separately from the official sources:
- RAF-DB: http://www.whdeng.cn/raf/model1.html
- FERPlus: https://github.com/microsoft/FERPlus
- AffectNet: http://mohammadmahoor.com/affectnet/
- SFEW 2.0: https://cs.anu.edu.au/few/AFEW.html

### For Evaluation

The evaluation datasets are already included in their respective evaluation folders:
- `eval_rafdb/data/rafdb_qa.json`
- `eval_ferplus/data/ferplus_qa.json`
- `eval_affectnet/data/affectnet_qa.json`
- `eval_sfew_2.0/data/sfew_2.0_qa.json`

The copies in this `data/` folder are provided for convenience.

## References

For more information about the dataset formats and usage, please refer to:
- Paper: https://arxiv.org/abs/2511.00389
- Model: https://huggingface.co/Karl28/UniFER-7B
- Dataset: https://huggingface.co/datasets/Karl28/UniFER

## Notes

The template files for `UniFER_CoT_230K.json` and `UniFER_RLVR_360K.json` contain sample entries demonstrating the expected format. For the full datasets (230K and 360K samples respectively), please download from the HuggingFace dataset repository.
