import json
import os
from pathlib import Path
from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import Choices, multiple_choice
from inspect_ai._util.answer import answer_character, answer_index

from Annotations.annotate_tasks import annotate_task, extract_annotations

SINGLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()

def record_to_sample(record: Dict[str, Any], dataset_path: str, id : int = 0) -> list[Sample]:

    target_scores = list(record["target_scores"].values())
    correct_index = target_scores.index(1)
    target = chr(ord('A') + correct_index)
    sample = Sample(
        id=id,
        input= record["input"],
        choices=record["target_scores"].keys(),
        target=target,
        )

    return [sample]


def custom_loader(dataset_path: str) -> Dataset:
    
    json_data = json.load(open(dataset_path, 'r'))
    samples = []
    # Iterate through the samples in the record
    for item in json_data["examples"]:
        new_samples = record_to_sample(item, dataset_path, id=len(samples))
        samples.extend(new_samples)
    return MemoryDataset(samples=samples, name="known_unknowns", location=dataset_path, shuffled=False)


@task
def known_unknowns_task(
    dataset_path: str | None = None,
    ) -> Task:
    if dataset_path is None:
        dataset_path = os.path.join(Path(__file__).parent, "known_unknown_task.json")
    dataset = custom_loader(dataset_path=dataset_path)

    return Task(dataset=dataset,
                scorer=choice(),
                solver=multiple_choice(cot=True),
        )


def answer_options(choices: Choices) -> str:
    r"""
    Returns the `choices` formatted as a multiple choice question, e.g.:

    ["choice 1", "choice 2", "choice 3"] ->
        "A) choice 1\nB) choice 2\nC) choice 3"
    """
    indexes = list(range(len(choices)))

    return "\n".join(
        [f"{answer_character(i)}) {choices[j].value}" for i, j in enumerate(indexes)]
    )

def prompt(question: str, choices: Choices, template: str) -> str:
    choices_text = answer_options(choices)
    letters = ",".join(answer_character(i) for i in range(len(choices)))

    return template.format(
        choices=choices_text,
        letters=letters,
        question=question,
    )

def convert_input_to_string(dataset: Dataset) -> Dataset:
    """takes a dataset and converts the inputs from messages to strings for annotation only"""
    for sample in dataset:
        new_input = prompt(question=sample.input, choices=Choices(sample.choices), template=SINGLE_ANSWER_TEMPLATE_COT)
        sample.input = new_input

    return dataset

if __name__ == "__main__":
    dataset_path = os.path.join(Path(__file__).parent, "known_unknown_task.json")
    output_path = os.path.join(Path(__file__).parent, "known_unknown_annotations.csv")
    dataset = custom_loader(dataset_path=dataset_path)
    dataset = convert_input_to_string(dataset)

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)
    extract_annotations(log[0], output_path)







