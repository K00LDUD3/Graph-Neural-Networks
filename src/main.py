import torch
import networkx as nx
from exprim import anchor, echo

#_: Echo Logger
logger = echo.Logger(sinks=[echo.ConsoleSink()])

#_: assembling config
from config import DeviceConfig
from dataclasses import dataclass, field
#@dataclass(frozen=True)
#class Config(anchor.BaseConfig):
#    device: Device = field(default_factory=Device)

def load_karate_club(logger: echo.Logger | None = None):

    G = nx.karate_club_graph()
    num_nodes = G.number_of_nodes()
    if logger:
        logger.info(f"num nodes: {num_nodes}")

    sources = []
    targets = []
    for edge in G.edges():
        u, v = edge
        sources.extend([u, v])
        targets.extend([v, u])

    #_: [2, 2 * num_edges] -> [2, 156] for faster indexing
    edge_index = torch.tensor([sources,targets], dtype=torch.long, device = DeviceConfig.cuda)
    
    #_: 34x34 identity matrix
    X = torch.eye(num_nodes, dtype=torch.float32, device=DeviceConfig.cuda)
    
    #_: label index mapping
    communities = [G.nodes[i]['club'] for i in range(num_nodes)]
    unique_clubs = list(set(communities))
    
    if logger:
        msg = ""
        for c in unique_clubs:
            msg += c + " "
        logger.info(msg)
    
    label_map = {club: idx for idx,club in enumerate(unique_clubs)}

    Y = torch.tensor([label_map[club] for club in communities], dtype=torch.long, device=DeviceConfig.cuda)

    return X, Y, num_nodes, edge_index


X, Y, num_nodes, edge_index = load_karate_club(logger)
print(f"feature shape(X): {X.shape}")
print(f"edge index shape: {edge_index.shape}")
print(f"labels: {Y.shape}")
