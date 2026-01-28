import spacy
import matplotlib.pyplot as plt
import networkx as nx


nlp = spacy.load("en_core_web_sm")


sentence = "naked running is forbidden"
doc = nlp(sentence)


edges = []
for token in doc:
    if token.dep_ == 'ROOT':
        edges.append((token.text, 'ROOT', token.text))
    else:
        edges.append((token.head.text, token.dep_, token.text))


G = nx.DiGraph()
for head, dep, child in edges:
    G.add_edge(head, child, label=dep)


def hierarchy_pos(G, root=None, width=0.8, vert_gap=0.5, vert_loc=0, xcenter=0.5):
    pos = {}
    def _hierarchy_pos(G, node, left, right, vert_loc, pos, parent=None):
        pos[node] = ((left + right) / 2, vert_loc)
        neighbors = list(G.neighbors(node))
        if parent in neighbors:
            neighbors.remove(parent)
        if neighbors:
            dx = (right - left) / len(neighbors)
            nextx = left
            for neighbor in neighbors:
                next_right = nextx + dx
                pos = _hierarchy_pos(G, neighbor, nextx, next_right, vert_loc - vert_gap, pos, node)
                nextx += dx
        return pos
    return _hierarchy_pos(G, root, 0, width, vert_loc, pos)

root = [t.text for t in doc if t.dep_ == 'ROOT'][0]
pos = hierarchy_pos(G, root=root, width=0.8, vert_gap=0.5)


plt.figure(figsize=(8, 6))
nx.draw(
    G, pos, with_labels=True,
    node_size=3000, node_color='lightyellow',
    font_size=14, font_weight='bold', arrows=True
)


edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=edge_labels,
    font_color='darkgreen', font_size=12
)

plt.title(f"Dependency Tree: '{sentence}'", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("naked.pdf")
