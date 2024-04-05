import io
import json
import logging
import os
import re
from functools import partial
from multiprocessing import Pool
from random import sample
from string import punctuation
from time import time

import numpy as np
from llama_index.llms.mistralai import MistralAI
from rouge_score import rouge_scorer
from tqdm import tqdm


class SelfInstruct:
    def __init__(
        self,
        api_key=None,
        model_name="mistral-medium",
        prompt_template_path="./prompt.txt",
        seed_tasks_path="./seed_tasks.jsonl",
        total_instructions=100,
        seed_sample_size=3,
        num_cpus=8,
    ):
        """SelfInstruct class."""
        self.prompt_template = open(prompt_template_path).read()
        self.seed_tasks_path = seed_tasks_path
        self.total_instructions = total_instructions
        self.seed_sample_size = seed_sample_size
        self.num_cpus = num_cpus
        self.llm = MistralAI(
            api_key=api_key,
            model=model_name,
            random_seed=42,
            temperature=0.7,
            max_tokens=3092,
            additional_kwargs={
                "top_p": 0.1,
            },
        )

    def generate(self, out_dir="./"):
        seed_tasks = [json.loads(l) for l in open(self.seed_tasks_path, "r")]
        seed_instruction_data = [
            {"instruction": t["instruction"], "response": t["response"]}
            for t in seed_tasks
        ]

        print(f"Loaded {len(seed_instruction_data)} seed instructions")

        os.makedirs(out_dir, exist_ok=True)
        request_idx = 0
        machine_instruction_data = []

        if os.path.exists(os.path.join(out_dir, "regen.json")):
            machine_instruction_data = self.jload(os.path.join(out_dir, "regen.json"))
            print(
                f"Loaded {len(machine_instruction_data)} machine-generated instructions"
            )

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        progress_bar = tqdm(total=self.total_instructions)

        if machine_instruction_data:
            progress_bar.update(len(machine_instruction_data))

        all_instructions = [d["instruction"] for d in seed_instruction_data] + [
            d["instruction"] for d in machine_instruction_data
        ]
        all_instruction_tokens = [
            scorer._tokenizer.tokenize(inst) for inst in all_instructions
        ]

        while len(machine_instruction_data) < self.total_instructions:
            request_idx += 1

            # only sampling from the seed tasks
            prompt_instructions = sample(seed_instruction_data, self.seed_sample_size)
            prompt = self.encode_prompt(prompt_instructions)
            
            request_start = time()
            result = self.mistral_completion(prompt)
            request_duration = time() - request_start

            process_start = time()
            instruction_data = self.post_process(self.seed_sample_size, result)

            total = len(instruction_data)
            keep = 0

            for entry in instruction_data:
                # compute similarity with the pre-tokenzied instructions
                new_instruction_tokens = scorer._tokenizer.tokenize(
                    entry["instruction"]
                )

                with Pool(self.num_cpus) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, new_instruction_tokens),
                        all_instruction_tokens,
                    )

                rouge_scores = [score.fmeasure for score in rouge_scores]
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }

                print(f"max: {max(rouge_scores)}")
                if max(rouge_scores) > 0.8:
                    print("skipping...")
                    continue
                else:
                    print("keeping!")
                    keep += 1

                entry["most_similar_instructions"] = most_similar_instructions
                entry["avg_similarity_score"] = float(np.mean(rouge_scores))

                machine_instruction_data.append(entry)

                all_instructions.append(entry["instruction"])
                all_instruction_tokens.append(new_instruction_tokens)

                progress_bar.update(1)

            process_duration = time() - process_start

            print(
                f"\nRequest {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s"
            )
            print(f"Generated {total} instructions, kept {keep} instructions")

            self.jdump(machine_instruction_data, os.path.join(out_dir, "regen.json"))
        else:
            print("\nGeneration complete!")

    def post_process(self, seed_sample_size, response):
        if response is None:
            return []

        raw_instructions = f"{seed_sample_size + 1}. Instruction: " + str(response)

        raw_instructions = re.split("\n\n", raw_instructions)
        instructions = []

        for idx, inst in enumerate(raw_instructions):
            idx += seed_sample_size + 1
            splitted_data = re.split(f"{idx}\.\s+(Instruction|Response):", inst)

            if len(splitted_data) != 5:
                continue
            else:
                inst = splitted_data[2].strip()
                output = splitted_data[4].strip()

            # filter out too short or too long instructions
            if len(inst.split()) < 3:
                continue

            # if last output is incomplete aka not end with a punctuation, skip
            if idx == len(raw_instructions) - 1 and output.split()[-1].endswith("."):
                continue

            # filter those starting with punctuation
            if inst[0] in punctuation:
                continue

            # filter those starting with non-english character
            if not inst[0].isascii():
                continue

            instructions.append(
                {"instruction": inst, "context": "", "response": output}
            )
            print({"instruction": inst, "context": "", "response": output})
        return instructions

    def mistral_completion(self, prompt):
        while True:
            try:
                completion = self.llm.complete(prompt)
                break
            except Exception as e:
                logging.warning(f"Error: {e}.")

        return completion

    def encode_prompt(self, prompt_instructions):
        """Encode multiple prompt instructions into a single string."""
        prompt = self.prompt_template
        last_idx = 0

        for idx, task_dict in enumerate(prompt_instructions):
            instruction, output = task_dict["instruction"], task_dict["response"]
            instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
            prompt += f"\n{idx + 1}. Instruction: {instruction}\n"
            prompt += f"{idx + 1}. Response: {output}\n"
            last_idx = idx


        prompt += f"\n{last_idx + 2}. Instruction: "

        return prompt

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode)
        return f

    def _make_w_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f_dirname = os.path.dirname(f)
            if f_dirname != "":
                os.makedirs(f_dirname, exist_ok=True)
            f = open(f, mode=mode)
        return f

    def jload(self, f, mode="r"):
        """Load a .json file into a dictionary."""
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    def jdump(self, obj, f, mode="w", indent=4, default=str):
        """
        Dump a str or dictionary to a file in json format.

        Args:
            obj: An object to be written.
            f: A string path to the location on disk.
            mode: Mode for opening the file.
            indent: Indent for storing json dictionaries.
            default: A function to handle non-serializable entries; defaults to `str`.
        """
        f = self._make_w_io_base(f, mode)
        if isinstance(obj, (dict, list)):
            json.dump(obj, f, indent=indent, default=default)
        elif isinstance(obj, str):
            f.write(obj)
        else:
            raise ValueError(f"Unexpected type: {type(obj)}")
        f.close()
