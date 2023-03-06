import cppast
import nx_pyg

def txt_to_pyg_data(input_filename):
    ast_graph = cppast.txt_to_nx_graph(input_filename)
    pyg_data = nx_pyg.to_pyg(ast_graph)
    return pyg_data


if __name__ == '__main__':
    input_filename = 'simple.txt'
    pyg_data = txt_to_pyg_data(input_filename)
    print(pyg_data)