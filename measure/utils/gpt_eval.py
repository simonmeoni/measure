import asyncio
import copy
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import openai
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()


def load_or_convert_to_dataframe(dataset_path: str) -> List[Dict]:
    """Load data from various file formats"""
    if "jsonl" in dataset_path:
        dataset = [json.loads(line) for line in open(dataset_path, "r")]
    elif "json" in dataset_path:
        with open(dataset_path, "r") as file:
            dataset = json.load(file)
    elif "parquet" in dataset_path:
        dataset = pd.read_parquet(dataset_path)
        dataset["response"] = dataset["response"].apply(lambda x: x.split("#")[0])
        dataset = dataset.to_dict("records")
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .json, .jsonl, or .parquet file."
        )
    return dataset


class GPTEvaluator:
    """GPT-based evaluation following the privacy.py philosophy"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.system_prompt = (
            "You are a helpful instruction-following assistant that prints "
            "the best model by selecting the best outputs for a given instruction."
        )

    def load_evaluation_prompt(self, prompt_path: str = None) -> str:
        """Load evaluation prompt template"""
        if prompt_path is None:
            prompt_path = os.path.join(
                os.path.dirname(__file__), "alpaca_eval_chat_gpt.txt"
            )

        with open(prompt_path, "r") as f:
            return f.read() + "\n"

    def _make_prompt(self, template: str, val_dict: Dict[str, str]) -> str:
        """Fill prompt template with values"""
        text_to_format = re.findall(r"{([^ \s]+?)}", template)
        prompt = copy.deepcopy(template)
        for to_format in text_to_format:
            prompt = prompt.replace("{" + to_format + "}", val_dict[to_format], 1)
        return prompt

    def _encode_prompt(
        self,
        prompt_template: str,
        instruction: str,
        model_output: Dict,
        reference_output: Dict,
        reference_first: bool = False,
    ) -> tuple:
        """Encode prompt with model outputs"""
        if reference_first:
            output_list = [reference_output, model_output]
        else:
            output_list = [model_output, reference_output]

        mapping_dict_output = {"instruction": instruction}
        mapping_dict_generator = {}

        for idx in range(2):
            mapping_dict_output["output_" + str(idx + 1)] = output_list[idx]["output"]
            mapping_dict_generator["model_" + str(idx + 1)] = output_list[idx][
                "generator"
            ]

        filled_prompt = self._make_prompt(prompt_template, mapping_dict_output)
        return filled_prompt, mapping_dict_generator

    async def _dispatch_requests(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 100,
        top_p: float = 1.0,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        timeout_seconds: int = 10,
        base_wait_time: float = 5,
        backoff_factor: float = 1.5,
    ) -> List[Any]:
        """Dispatch async requests to OpenAI API with retry logic"""

        async def send_request(message):
            return await self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

        async def request_until_success(message):
            wait_time = base_wait_time
            while True:
                try:
                    return await asyncio.wait_for(
                        send_request(message), timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    print(f"Timeout! Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    wait_time *= backoff_factor
                except openai.BadRequestError as e:
                    print(f"Bad request error: {str(e)}, message: {message}")
                    return None

        async_responses = [request_until_success(x) for x in messages_list]
        return await asyncio.gather(*async_responses)


def gpt_evaluate(
    model_data: Union[str, List[Dict], pd.DataFrame],
    reference_data: Union[str, List[Dict], pd.DataFrame],
    model_name: str = "model",
    reference_name: str = "reference",
    task_name: str = "alpaca_eval",
    reference_first: bool = False,
    batch_size: int = 20,
    max_samples: int = None,
    output_dir: str = None,
    resume: bool = True,
) -> Dict[str, float]:
    """
    Main GPT evaluation function following privacy.py philosophy

    Args:
        model_data: Model outputs to evaluate (file path or data)
        reference_data: Reference outputs to compare against (file path or data)
        model_name: Name of the model being evaluated
        reference_name: Name of the reference model
        task_name: Task type (alpaca_eval, medsi, iCliniq)
        reference_first: Whether to present reference output first
        batch_size: Batch size for API requests
        max_samples: Maximum number of samples to evaluate
        output_dir: Directory to save results (for resume capability)
        resume: Whether to resume from existing results

    Returns:
        Dictionary of flattened metrics
    """
    # Load data
    if isinstance(model_data, str):
        model_data = load_or_convert_to_dataframe(model_data)
    elif isinstance(model_data, pd.DataFrame):
        model_data = model_data.to_dict("records")

    if isinstance(reference_data, str):
        reference_data = load_or_convert_to_dataframe(reference_data)
    elif isinstance(reference_data, pd.DataFrame):
        reference_data = reference_data.to_dict("records")

    # Limit samples if specified
    if max_samples is not None and max_samples != -1:
        model_data = model_data[:max_samples]
        reference_data = reference_data[:max_samples]

    # Extract instructions based on task type
    if task_name == "alpaca_eval" or task_name == "medsi":
        instructions = [item["instruction"] for item in reference_data]
    elif task_name == "iCliniq":
        instructions = [item["instruction"] for item in model_data]
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    # Normalize output format
    def normalize_output(data, generator_name):
        normalized = []
        for item in data:
            if "response" in item:
                normalized.append(
                    {"generator": generator_name, "output": item["response"]}
                )
            else:
                normalized.append(
                    {"generator": generator_name, "output": item["output"]}
                )
        return normalized

    model_outputs = normalize_output(model_data, model_name)
    reference_outputs = normalize_output(reference_data, reference_name)

    # Ensure same length
    min_len = min(len(instructions), len(model_outputs), len(reference_outputs))
    instructions = instructions[:min_len]
    model_outputs = model_outputs[:min_len]
    reference_outputs = reference_outputs[:min_len]

    total = len(instructions)

    # Setup output file for resume capability
    existing_results = []

    # Initialize evaluator and progress bar
    evaluator = GPTEvaluator()
    prompt_template = evaluator.load_evaluation_prompt()
    progress_bar = tqdm(total=total)
    progress_bar.update(len(existing_results))

    # Process remaining samples
    idx = len(existing_results)
    all_results = existing_results.copy()

    try:
        while idx < total:
            # Prepare batch
            message_list = []
            model2name_list = []
            batch_end = min(idx + batch_size, total)

            for j in range(idx, batch_end):
                task_prompt, model2name = evaluator._encode_prompt(
                    prompt_template,
                    instructions[j],
                    model_outputs[j],
                    reference_outputs[j],
                    reference_first,
                )
                message = [
                    {"role": "system", "content": evaluator.system_prompt},
                    {"role": "user", "content": task_prompt},
                ]
                message_list.append(message)
                model2name_list.append(model2name)

            # Process batch with retry logic
            batch_results = []
            target_length = 100
            retry_cnt = 0
            wait_base = 10.0

            while len(batch_results) == 0:
                try:
                    batch_predictions = asyncio.run(
                        evaluator._dispatch_requests(
                            messages_list=message_list,
                            temperature=0,
                            max_tokens=target_length,
                            top_p=1.0,
                            frequency_penalty=0,
                            presence_penalty=0,
                        )
                    )

                    for j, message in enumerate(message_list):
                        if batch_predictions[j] is not None:
                            data = {
                                "prompt": message[1]["content"],
                                "response": batch_predictions[j]
                                .choices[0]
                                .message.content,
                            }
                            batch_results.append(data)

                    retry_cnt = 0
                    break

                except openai.OpenAIError as e:
                    print(f"OpenAIError: {e}.")
                    if "Please reduce the length of the messages or completion" in str(
                        e
                    ):
                        target_length = int(target_length * 0.8)
                        print(f"Reducing target length to {target_length}, retrying...")
                    else:
                        retry_cnt += 1
                        print("retry number: ", retry_cnt)
                        time.sleep(wait_base)
                        wait_base = wait_base * 1.5

            # Save results
            for model2name, result in zip(model2name_list, batch_results):
                merged_dict = {**model2name, **result}
                all_results.append(merged_dict)
                progress_bar.update(1)
                idx += 1

    finally:
        progress_bar.close()

    # Parse results and return both metrics and raw results
    return all_results
