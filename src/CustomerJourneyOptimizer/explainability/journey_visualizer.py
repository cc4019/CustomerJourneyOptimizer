import matplotlib.pyplot as plt
import networkx as nx

class JourneyVisualizer:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_transition(self, from_state, to_state, weight=1):
        """
        Add a transition between states in the journey.
        
        :param from_state: Starting state
        :param to_state: Ending state
        :param weight: Weight of the transition
        """
        self.G.add_edge(from_state, to_state, weight=weight)

    def visualize(self):
        """
        Visualize the customer journey as a directed graph.
        """
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=10, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        
        plt.title("Customer Journey Visualization")
        plt.axis('off')
        plt.show()
