from graphviz import Digraph, Graph

class GraphVisualisation():

    def __init__(self, max_level = float('inf')):
        self.max_level = max_level

    def single_agent_mcts_to_graph(self, node, filename='mcts'):
        g = Digraph('G', filename=filename, format='png')
        self.single_agent_node_to_graph(g, node, level = 0)
        return g

    def node_label(self, node):
        return "V%s =\n%0.2f\\nN = %d" % (node.state, node.get_value(), node.get_visits())

    def single_agent_node_to_graph(self, g, node, level):
        g.node(str(node.id), label=self.node_label(node), **{'width':str(1), 'height':str(1), 'fixedsize':str(True)})
        for action in node.children.keys():
            g.edge(str(node.id), str(action) + str(node.id), action)

        if level <= self.max_level:
            for action in node.children.keys():
                self.environment_node(g, node, action, level)

    def environment_node(self, g, node, action, level):
        node_id = str(action) + str(node.id)
        for (child, probability) in node.children[action]:
            g.node(node_id, node_id, style='filled', shape='point', width='0.25')
            g.edge(node_id, str(child.id), str(probability))

        for (child, _) in node.children[action]:
            self.single_agent_node_to_graph(g, child, level + 1)

    def node_to_graph(self, game, node, filename='backward_induction', print_state = False, print_value = False):
        graph = Graph('G', filename=filename, format='png')
        self.game_node(graph, game, node, visited = [], level = 1, print_state = print_state, print_value = print_value)
        return graph

    def node_to_string(self, game, node, print_state, print_value):
        result = ""
        if print_state:
            result += game.to_string(node.state) + "\\n"
        if len(node.children) == 0 or print_value:
            result += "("
            result += ", ".join([str(node.value[player]) for player in node.value.keys()])
            result += ")"
        return result

    def game_node(self, graph, game, node, visited, level, print_state, print_value):
        if node.id not in visited:
            graph.node(str(node.id), label=self.node_to_string(game, node, print_state, print_value), xlabel = str(node.player_turn) if node.player_turn is not None else "")
            if level <= self.max_level:
                for key in node.children.keys():
                    child = node.children[key]
                    self.game_node(graph, game, child, visited, level + 1, print_state, print_value)
                    penwidth = '3.0' if child.is_best_action else '1.0'
                    graph.edge(str(node.id), str(child.id), str(key), penwidth = penwidth)
            visited += [node.id]
            


