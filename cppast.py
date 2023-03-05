import clang.cindex
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import graphviz
import tempfile
import matplotlib.pyplot as plt

# create an index
clang.cindex.Config.set_library_file('/Library/Developer/CommandLineTools/usr/lib/libclang.dylib')
index = clang.cindex.Index.create()

input_filename = 'simple.txt'
output_filename = 'output.cpp'

with open(input_filename, 'r') as input, open(output_filename, 'w') as output:
    for line in input:
        output.write(line)

root = index.parse(output_filename)

# A dict to show label attributes. 
label_dict = {}

def traverse_ast(node, parent=None, graph=None, first=False):
    # TODO: the "first" parameter should be deleted
    # if node.kind.name == "INTEGER_LITERAL":
    #     print(dir(node))
    #     print(dir(node.kind))
    #     print(list(node.get_tokens())[0].spelling)


    if graph is None:
        graph = nx.DiGraph()
    if node.kind.is_unexposed():
        # Unexposed nodes are compiler specific and should be skipped over
        for child in node.get_children():
            traverse_ast(child, parent, graph)
    else:
        # Add the current node to the graph
        node_id = str(node.hash)
        node_label = node.kind.name
        # maybe need to make sure node label is 7 chars long 
        # This part is also here for considering literal values, but should it be?
        if str(node_label)[-7:] == "LITERAL":
            # Differentiating string literals and other numeric literals, who knows if this is a good idea or not
            if node_label[:6] == "STRING":
                # This line is here only for visualization purposes. If we are running some algo on this graph then don't do this
                # node_label += ' ' + str(node.spelling)

                graph.add_node(node_id, label=node_label, string_val = node.spelling)
            else:
                graph.add_node(node_id, label=node_label, literal_val = next(node.get_tokens()).spelling)
            
        else:
            graph.add_node(node_id, label=node_label)

        label_dict[node_id] = node_label
        # Add an edge from the current node to its parent
        if parent is not None:
            graph.add_edge(node_id, parent)
        # Recursively traverse the children of the current node
        for child in node.get_children():
            traverse_ast(child, node_id, graph)
    return graph

# Traverse the AST and construct the graph
ast_graph = traverse_ast(root.cursor, first=True)
nx.draw(ast_graph, labels=label_dict, with_labels = True)
plt.show()
print(f"The graph has {len(ast_graph.nodes)} nodes")
print(f"The graph has {len(ast_graph.edges)} edges")
# agraph = to_agraph(ast_graph)
# agraph.draw('sample_ast.png', prog='sfdp')
