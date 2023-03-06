import clang.cindex
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import graphviz
import tempfile
import matplotlib.pyplot as plt

import nx_pyg as nf

# create an index
clang.cindex.Config.set_library_file('/Library/Developer/CommandLineTools/usr/lib/libclang.dylib')
index = clang.cindex.Index.create()

OUTPUT_FILENAME = 'output.cpp'

def get_root(input_filename, output_filename=OUTPUT_FILENAME):
    with open(input_filename, 'r') as input, open(output_filename, 'w') as output:
        for line in input:
            output.write(line)

    root = index.parse(output_filename)
    return root

def traverse_ast(node, parent=None, graph=None, first=False, label_dict=None):
    # TODO: the "first" parameter should be deleted
    # TODO: review and clean up function
    # if node.kind.name == "INTEGER_LITERAL":
    #     print(dir(node))
    #     print(dir(node.kind))
    #     print(list(node.get_tokens())[0].spelling)

    if graph is None:
        graph = nx.DiGraph()
    if node.kind.is_unexposed():
        # Unexposed nodes are compiler specific and should be skipped over
        for child in node.get_children():
            traverse_ast(child, parent, graph, label_dict=label_dict)
    else:
        # Add the current node to the graph
        node_id = str(node.hash)
        node_label = node.kind.name
        # print("node_type: ", node_type)
        # maybe need to make sure node label is 7 chars long 
        # This part is also here for considering literal values, but should it be?
        if str(node_label)[-7:] == "LITERAL":
            # Differentiating string literals and other numeric literals, who knows if this is a good idea or not
            if node_label[:6] == "STRING":
                # This line is here only for visualization purposes. If we are running some algo on this graph then don't do this
                # node_label += ' ' + str(node.spelling)

                graph.add_node(node_id, node_label=nf.label_to_categorical(node_label), node_val = node.spelling)
            else:
                try:
                    val = next(node.get_tokens()).spelling
                except StopIteration:
                    val = None
                graph.add_node(node_id, node_label=nf.label_to_categorical(node_label), node_val = val)
            
        else:
            # Do we want to add the spelling of non-literals to the graph?
            graph.add_node(node_id, node_label=nf.label_to_categorical(node_label), node_val = '')

        if isinstance(label_dict, dict):
            label_dict[node_id] = node_label

        # Add an edge from the current node to its parent
        if parent is not None:
            graph.add_edge(node_id, parent)
        # Recursively traverse the children of the current node
        for child in node.get_children():
            traverse_ast(child, node_id, graph, label_dict=label_dict)
    return graph


def txt_to_nx_graph(input_filename, output_filename=OUTPUT_FILENAME):
    root = get_root(input_filename, output_filename)
    graph = traverse_ast(root.cursor)
    return graph


if __name__ == '__main__':
    input_filename = 'simple.txt'
    root = get_root(input_filename)

    # Traverse the AST and construct the graph
    label_dict = {}
    ast_graph = traverse_ast(root.cursor, first=True, label_dict=label_dict)
    print(f"The graph has {len(ast_graph.nodes)} nodes")
    print(f"The graph has {len(ast_graph.edges)} edges")

    data = nf.to_pyg(ast_graph)
    print("PyG Data: ", data)

    nx.draw(ast_graph, labels=label_dict, with_labels = True)
    plt.show()