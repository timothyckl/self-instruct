## Install Dependencies

This repository contains a simple dataset generation script following [Self-Instruct](https://arxiv.org/abs/2212.10560) with Mistral models.

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


```

## Notes

- Modification of the prompt template or formatting will require appropriate changes in class methods for your use case.
- This script uses Mistral as an example, it can be changed to use other models as well.
