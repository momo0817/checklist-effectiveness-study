# Are Checklists Really Useful for Automatic Evaluation of Generative Tasks?
---
We investigate whether checklists should be used for all questions or selectively.
We generate checklist via six methods and evakuate effectiveness across eigit model sizes, and identify checklist items correlated with human evaluations.

- Paper: Are Checklists Really Useful for Automatic Evaluation of Generative Tasks?
---
# Installation
```
git clone https://github.com/princeton-nlp/LLMBar.git
pip install -r requirements.txt
```
We also use InFoBench's human evaluations.
Please download the dataset available on their GitHub.
InFoBench's GitHub: https://github.com/qinyiwei/InfoBench?tab=readme-ov-file

## Environment Variable
---
```
export HF_TOKEN=YOUR_HUGGINGFACE_ACCESS_TOKEN
export HF_CACHE_DIR=DIR_NAME
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base
```

## Create Subset
### LLMBar

```
python3 src/sample_llmbar.py --llmbar-dir ./LLMBar/Dataset/ --output-dir ./LLMBar/Dataset/Sample
```
### InFoBench
```
python3 src/sample_infobench.py --infobench-dir ./InFoBench/expert_annotation --output-dir ./data/dataset/InFoBench/expert_annotation
```

## Experiment
---
We can generate checklist and evaluate response using bellow shell.
```
./example/experiment.sh <checklist_model_name> <eval_model_name> <dataset_sub_dir> <checklist_type>
```
-  checklist_model_name: to generate checklist model 
-  eval_model_name: to evaluate response model 
-  dataset: dataset sub dir
-  checklist_type: we choose six method
    - baseline: our baseline
    - specify: 
    - refine_baseline:
    - checklist_length_0.5: 
    - checklist_length_1.5: 
    - ticking: 

### Example 1 (Dataset type: Pairwise comparison, Prompt: Baseline or Ticking or Specify)
```
./example/normal_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 specify
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Dataset: Sample
- Checklist type: specify

### Example 2 (Dataset type: Pairwise comparison, Prompt: Self-refine)
```
./example/refine_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 Sample baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Dataset: Sample
- Checklist type: baseline

### Example 3 (Dataset type: Pairwise comparison, Prompt: checklist Length (0.5, 1.5))
```
./example/adjust_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 Sample baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Dataset: Sample
- Checklist type: baseline


### Example 4 (Dataset type: Direct Scoring, Prompt: Baseline or Ticking or Specify)
```
./example/abs_normal_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 InFoBench/expert_annotation baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Dataset: Sample
- Checklist type: baseline

### Example 5 (Dataset type: Direct Scoring, Prompt: Self-refine)
```
./example/abs_refine_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 Sample baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Dataset: Sample
- Checklist type: baseline

### Example 6 (Dataset type: Direct Scoring, Prompt: Checklist Length (0.5, 1.5))
```
./example/abs_adjust_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 Sample baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Dataset: Sample
- Checklist type: baseline


---

## Analysis

### Pairwise Comparison Dataset


### Direct Scoring Dataset


## Citation
```

```


