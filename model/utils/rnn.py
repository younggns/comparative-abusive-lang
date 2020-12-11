import datetime
from typing import List

from ..layers.RNN_process_data import ProcessData


class RNNUtil:

    def saveResult(
            self: RNNUtil,
            batch_gen: ProcessData,
            graph_dir_name: str,
            accr_class: List,
            recall_class: List,
            f1_class: List,
            accr_avg,
            recall_avg,
            f1_avg):
        with open('./TEST_run_result.txt', 'a') as f:
            f.write(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' +
                batch_gen.data_path.split('/')[-2] + '\t' +
                graph_dir_name + '\t' + \
                # str(best_dev_f1) + '\t' + \
                # str(test_f1_at_best_dev) + '\t' + \
                str(accr_class[0]) + '\t' + \
                str(recall_class[0]) + '\t' + \
                str(f1_class[0]) + '\t\t' + \
                str(accr_class[1]) + '\t' + \
                str(recall_class[1]) + '\t' + \
                str(f1_class[1]) + '\t\t' + \
                str(accr_class[2]) + '\t' + \
                str(recall_class[2]) + '\t' + \
                str(f1_class[2]) + '\t\t' + \
                str(accr_class[3]) + '\t' + \
                str(recall_class[3]) + '\t' + \
                str(f1_class[3]) + '\t\t' + \
                str(accr_avg) + '\t' + \
                str(recall_avg) + '\t' + \
                str(f1_avg) + \
                '\n'
            )
