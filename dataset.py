import cppast
import nx_pyg

import os
import json
import torch

import shutil

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


def json_to_pyg_dataset(input_filename, output_filename, stop_at=100):
    pyg_dataset = []
    with open(input_filename, 'r') as f:
        for line in f:
            item = json.loads(line)
            tmp_filename = 'tmp.txt'
            with open(tmp_filename, 'w') as f:
                f.write(item['code'])
            pyg_data = txt_to_pyg_data(tmp_filename, int(item['label']), int(item['index']))
            pyg_dataset.append(pyg_data)
            if len(pyg_dataset) == stop_at:
                break
    torch.save(pyg_dataset, output_filename)
    print(len(pyg_dataset))

def zip_datasets(dir_name, output_filename):
    shutil.make_archive(output_filename, 'zip', dir_name)



if __name__ == '__main__':
    # json_to_pyg_dataset('Clone-detection-POJ-104/dataset/train.jsonl', 'data/poj-104/train.pt')
    # json_to_pyg_dataset('Clone-detection-POJ-104/dataset/valid.jsonl', 'data/poj-104/valid.pt')
    # json_to_pyg_dataset('Clone-detection-POJ-104/dataset/test.jsonl', 'data/poj-104/test.pt')
    # zip_datasets('data/poj-104', 'data/poj-104')
    json_to_pyg_dataset('Clone-detection-POJ-104/dataset/train.jsonl', 'data/poj-104-mini/train.pt', stop_at=1600)
    json_to_pyg_dataset('Clone-detection-POJ-104/dataset/valid.jsonl', 'data/poj-104-mini/valid.pt', stop_at=400)
    json_to_pyg_dataset('Clone-detection-POJ-104/dataset/test.jsonl', 'data/poj-104-mini/test.pt', stop_at=600)
    zip_datasets('data/poj-104-mini', 'data/poj-104-mini')

