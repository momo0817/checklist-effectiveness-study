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
python3 src/experiment/sample_llmbar.py --llmbar-dir ./LLMBar/Dataset/ --output-path ./Dataset/LLMBar/dataset.json
```
### InFoBench
```
python3 src/experiment/sample_infobench.py --infobench-dir ./InFoBench/expert_annotation --output-path ./Dataset/InFoBench/dataset.json
```

## Experiment
---
We can generate checklist and evaluate response using bellow shell.
```
./shell_script/experiment/**_experiment.sh <checklist_model_name> <eval_model_name>  <checklist_type>
```
-  checklist_model_name: to generate checklist model 
-  eval_model_name: to evaluate response model 
-  dataset: dataset sub dir
-  checklist_type: we choose six method
    - baseline: our baseline
    - specify: we add to the baseline that checklist questions should be designed considering possible responses to the input.
    - refine_baseline:  a Likert scale evaluation and accompanying feedback based on the baseline prompt and uses this feedback to regenerate improved checklists.
    - adjust_0.5_baseline: Adjust the length of the baseline checklist to 0.5 times its original size.
    - adjust_1.5_baseline: Adjust the length of the baseline checklist to 1.5 times its original size.
    - ticking: . TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation's prompt 
     - their paper: https://arxiv.org/abs/2410.03608

### Example 1 (Dataset type: Pairwise comparison, Prompt: Baseline or Ticking or Specify)
```
./shell_script/experiment/normal_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 specify
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Checklist type: specify

### Example 2 (Dataset type: Pairwise comparison, Prompt: Self-refine)
```
./shell_script/experiment/refine_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Checklist type: baseline

### Example 3 (Dataset type: Pairwise comparison, Prompt: checklist Length (0.5, 1.5))
```
./shell_script/experiment/adjust_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Checklist type: baseline


### Example 4 (Dataset type: Direct Scoring, Prompt: Baseline or Ticking or Specify)
```
./shell_script/experiment/abs_normal_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Checklist type: baseline

### Example 5 (Dataset type: Direct Scoring, Prompt: Self-refine)
```
./shell_script/experiment/abs_refine_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Checklist type: baseline

### Example 6 (Dataset type: Direct Scoring, Prompt: Checklist Length (0.5, 1.5))
```
./shell_script/experiment/abs_adjust_experiment.sh gpt-4o-2024-08-06 gpt-4o-2024-08-06 baseline
```
- Checklist generaliton model: gpt-4o-2024-08-06
- Evaluation model: gpt-4o-2024-08-06
- Checklist type: baseline


---

## Analysis
We can generate checklist and evaluate response using bellow shell.
```
./shell_script/analysis/<dataset-type>.sh <checklist_model> <eval_model> <checklist_type>
```


### Pairwise Comparison Dataset
```
./shell_script/analysis/pairwise_analysis.sh gpt-4o-2024-08-06 Qwen/Qwen2.5-7B-Instruct baseline
```

### Direct Scoring Dataset
```
./shell_script/analysis/scoring_analysis.sh  gpt-4o-2024-08-06 Qwen/Qwen2.5-7B-Instruct baseline
```

## Citation
```
@article{furuhashi2025are,

  author = {Furuhashi Momoka, Nakayama Kouta, Kodama Takashi,and Sugawara Saku},

  title = {Are Checklists Really Useful for Automatic Evaluation of Generative Tasks?},

  booktitle = {The 2025 Conference on Empirical Methods in Natural Language Processing},

  year = {2025},


}

```


