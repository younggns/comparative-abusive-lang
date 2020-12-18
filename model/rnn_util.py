import datetime
from typing import List

from layers.RNN_process_data import ProcessData


def create_result(title, accuracy, recall, f1) -> str:
    return f"""
## {title}
Accuracy: {accuracy}
Recall: {recall}
f1: {f1}
"""


def save_result(
        batch_gen: ProcessData,
        graph_dir_name: str,
        accr_class: List,
        recall_class: List,
        f1_class: List,
        accr_avg,
        recall_avg,
        f1_avg):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    data = batch_gen.data_path.split('/')[-2]
    result_header = f"{time_now}\t{data}\t{graph_dir_name}"

    with open('./TEST_run_result.txt', 'a') as f:
        f.write(result_header)

        # Write results
        for index, accuracy, recall, f1 in enumerate(
                zip(accr_class, recall_class, f1_class)):
            f.write(create_result(
                title=index,
                accuracy=accuracy,
                recall=recall,
                f1=f1
            ))

        # Write average
        f.write(create_result(
            title='Average',
            accuracy=accr_avg,
            recall=recall_avg,
            f1=f1_avg
        ))
