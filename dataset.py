import cppast
import nx_pyg

import os
import json
import torch

def txt_to_pyg_data(input_filename, task_label, index):
    ast_graph = cppast.txt_to_nx_graph(input_filename)
    pyg_data = nx_pyg.to_pyg(ast_graph)
    pyg_data.task_label = task_label
    pyg_data.index = index
    return pyg_data

def get_file_paths(data_dir):
    file_paths = []
    for root, directories, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


def json_to_pyg_dataset(input_filename, output_filedir):
    # pyg_dataset = []
    i = 0
    with open(input_filename, 'r') as f:
        for line in f:
            item = json.loads(line)
            tmp_filename = 'tmp.txt'
            with open(tmp_filename, 'w') as f:
                f.write(item['code'])
            pyg_data = txt_to_pyg_data(tmp_filename, int(item['label']), int(item['index']))
            torch.save(pyg_data, f'{output_filedir}/{i}.pt')
            i += 1
    print(i)

if __name__ == '__main__':
    json_to_pyg_dataset('Clone-detection-POJ-104/dataset/train.jsonl', 'data/poj-104/train')
    json_to_pyg_dataset('Clone-detection-POJ-104/dataset/valid.jsonl', 'data/poj-104/valid')
    json_to_pyg_dataset('Clone-detection-POJ-104/dataset/test.jsonl', 'data/poj-104/test')

