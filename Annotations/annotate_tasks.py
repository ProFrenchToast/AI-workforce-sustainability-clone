


import json
import os
import csv
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.log import EvalLog

from Annotations.annotation_agent import annotation_agent


PROMPT_TEMPLATE = """QUERY: The following rubric describes six distinct levels of *{dimension}*
required by different tasks:
{rubric}
TASK INSTANCE: {instance}
INSTRUCTION: Score the level of *{dimension}* demanded by the given
TASK INSTANCE using a discrete value from 0 to 5. Use CHAIN-OF-THOUGHTS
REASONING to reason step by step before assigning the score. After the
CHAIN-OF-THOUGHTS REASONING STEPS, conclude your assessment with the
statement: "Thus, the level of *{dimension}* demanded by the given TASK
INSTANCE is: SCORE", where 'SCORE' is the integer score you have determined.
"""


def combine_dataset(task_dataset: Dataset, rubric_data) -> Dataset:
    new_samples = []
    for sample in task_dataset:
        instance = sample.input
        for rubric_dict in rubric_data:
            dimension = rubric_dict["dimension"]
            rubric = rubric_dict["rubric"]
            combined_string = PROMPT_TEMPLATE.format(dimension=dimension, rubric=rubric, instance=instance)
            new_sample = Sample(
                input=combined_string,
                id=f"{sample.id}_{dimension}",
                metadata={
                    "dimension": dimension,
                    "sample_id": sample.id,
                }
                )
            new_samples.append(new_sample)
    return MemoryDataset(new_samples, name=f"{task_dataset.name}_annotation")

def extract_annotations(log: EvalLog, output_file: str):
    assert log.samples is not None
    annotations = []
    for sample in log.samples:
        score = sample.output.completion
        annotation = (log.eval.dataset.name, sample.metadata["sample_id"], sample.metadata["dimension"], score)
        annotations.append(annotation)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dataset name", "sample id", "dimension", "score"])
        writer.writerows(annotations)




@task
def annotate_task(
    task_dataset: Dataset,
):
    assert task_dataset is not None
    rubric_data = json.load(open(
        os.path.join(Path(__file__).parent, "./rubric.json"), "r"))
    # now combine rubrics and samples in the template 
    annotation_dataset = combine_dataset(task_dataset, rubric_data)

    return Task(
        dataset=annotation_dataset,
        solver=annotation_agent(),
    )




