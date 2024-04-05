# Self-Instruct

This repository contains a simple dataset generation script following [Self-Instruct](https://arxiv.org/abs/2212.10560) with Mistral models.

## Install Dependencies

```bash
pip install -r requirement.txt
```

## Usage

To run the generation script:

```bash
python main.py
```

Additional options:

```bash
python main.py -h

# usage: main.py [-h] [-m MODEL] [-p PROMPT_TEMPLATE_PATH] [-s SEED_TASKS_PATH] [-g TOTAL_INSTRUCTIONS] [-z SEED_SAMPLE_SIZE] [-o OUTPUT_DIR]

# options:
#   -h, --help            show this help message and exit
#   -m MODEL, --model MODEL
#                         Name of mistral model
#   -p PROMPT_TEMPLATE_PATH, --prompt_template_path PROMPT_TEMPLATE_PATH
#                         Path to prompt template
#   -s SEED_TASKS_PATH, --seed_tasks_path SEED_TASKS_PATH
#                         Path to seed task file
#   -g TOTAL_INSTRUCTIONS, --total_instructions TOTAL_INSTRUCTIONS
#                         Total number of instructions to generate
#   -z SEED_SAMPLE_SIZE, --seed_sample_size SEED_SAMPLE_SIZE
#                         Number of seed tasks to sample during generation
#   -o OUTPUT_DIR, --output_dir OUTPUT_DIR
#                         Directory to store outputs
```

## Notes

- Modification of the prompt template or formatting will require appropriate changes in class methods for your use case.
- This script uses Mistral as an example, it can be changed to use other models as well.
