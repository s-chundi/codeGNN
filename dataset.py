import cppast
import nx_pyg

import os

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

if __name__ == '__main__':
    input_filename = 'simple.txt'
    pyg_data = txt_to_pyg_data(input_filename)
    print(pyg_data)

