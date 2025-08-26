

import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence

from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, accuracy, includes, mean, metric, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
import numpy as np

from Annotations.annotate_tasks import annotate_task, extract_annotations

def record_to_sample(record: Dict[str, Any], dataset_path: str) -> list[Sample]:
    sample_id = record["id"]
    context = record["story"]
    turn_id = 1
    messages: list[ChatMessage] = [ChatMessageUser(content=context)]
    samples = []

    for question in record["questions"]:
        assert question["turn_id"] == turn_id
        messages.append(ChatMessageUser(content=question["input_text"]))
        # first find the answers to this question
        current_answers = []

        for answer in record["answers"]:
            if answer["turn_id"] == question["turn_id"]:
                current_answers.append(answer["input_text"])
                golden_answer = answer["input_text"]

        for answer_set in record["additional_answers"].values():
            for answer in answer_set:
                if answer["turn_id"] == question["turn_id"]:
                    current_answers.append(answer["input_text"])
        # then create the sample
        sample = Sample(
            input=messages.copy(), 
            target=current_answers, 
            id=f"{sample_id}_{turn_id}",
            )
        samples.append(sample)

        # then add the "golden answer" to the messages and repeat
        messages.append(ChatMessageAssistant(content=golden_answer))
        turn_id += 1

    return samples

def custom_loader(dataset_path: str) -> Dataset:
    
    json_data = json.load(open(dataset_path, 'r'))
    samples = []
    # Iterate through the samples in the record
    for item in json_data["data"]:
        new_samples = record_to_sample(item, dataset_path)
        samples.extend(new_samples)
    return MemoryDataset(samples=samples, name="CoQA", location=dataset_path, shuffled=False)


@task
def CoQA_task(
    dataset_path: str | None = None,
    ) -> Task:
    if dataset_path is None:
        dataset_path = os.path.join(Path(__file__).parent, "coqa.test.json")
    dataset = custom_loader(dataset_path=dataset_path)

    return Task(dataset=dataset,
                scorer=includes(),
        )


def convert_input_to_string(dataset: Dataset) -> Dataset:
    """takes a dataset and converts the inputs from messages to strings for annotation only"""
    for sample in dataset:
        new_input = ""
        for message in sample.input:
            new_input += f"{message.role}: {message.content}\n"
        sample.input = new_input

    return dataset

if __name__ == "__main__":
    dataset_path = os.path.join(Path(__file__).parent, "coqa.test.json")
    output_path = os.path.join(Path(__file__).parent, "coqa_annotations.csv")
    dataset = custom_loader(dataset_path=dataset_path)
    dataset = convert_input_to_string(dataset)

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)
    extract_annotations(log[0], output_path)

