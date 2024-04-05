import os
from argparse import ArgumentParser
from self_instruct import SelfInstruct

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

parser = ArgumentParser()

parser.add_argument("-m", "--model", type=str, default="mistral-medium", help="Name of mistral model")
parser.add_argument("-p", "--prompt_template_path", type=str, default="./prompt.txt", help="Path to prompt template")
parser.add_argument("-s", "--seed_tasks_path", type=str, default="./seed-tasks.jsonl", help="Path to seed task file")
parser.add_argument("-g", "--total_instructions", type=int, default=100, help="Total number of instructions to generate")
parser.add_argument("-z", "--seed_sample_size", type=int, default=5, help="Number of seed tasks to sample during generation")
parser.add_argument("-o", "--output_dir", type=str, default="./ouputs", help="Directory to store outputs")

args = parser.parse_args()

if __name__ == "__main__":
    alg = SelfInstruct(
        api_key=MISTRAL_API_KEY,
        model_name=args.model,
        prompt_template_path=args.prompt_template_path,
        seed_tasks_path=args.seed_tasks_path,
        total_instructions=args.total_instructions,
        seed_sample_size=args.seed_sample_size,
    )

    alg.generate(out_dir=args.output_dir)
