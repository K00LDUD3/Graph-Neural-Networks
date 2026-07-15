import torch
import networkx as nx
from exprim import anchor, echo



#_: assembling config
from config import DeviceConfig
from dataclasses import dataclass, field
#@dataclass(frozen=True)
#class Config(anchor.BaseConfig):
#    device: Device = field(default_factory=Device)

#_: dataset loader
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

    if logger:
        logger.info(f"feature shape(X): {X.shape}")
        logger.info(f"edge index shape: {edge_index.shape}")
        logger.info("labels: {Y.shape}")
    
    return X, Y, num_nodes, edge_index

#_: self loops
def add_self_loops(edge_index: torch.Tensor, num_nodes: int):
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2,1)  # shape to [2,num_nodes]

    edge_index_with_loops = torch.cat([edge_index, loop_index], dim=1)
    return edge_index_with_loops
    
#_:
def compute_symmetric_normalization(edge_index, num_nodes):
    """
    """
    #WARN: assuming edges have weight of 1 each
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
    
    # row 0 = source, row 1 = target
    row, col = edge_index[0], edge_index[1]
    
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    deg.scatter_add_(0, row, edge_weight)  # Sums up occurrences of each source node
    
    # D^{-1/2}
    deg_inv_sqrt = torch.pow(deg, -0.5)
    # Handle isolated nodes if degree is 0 to avoid division by zero / inf
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    
    # d_i^{-1/2} * A_ij * d_j^{-1/2}
    norm_weights = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    return norm_weights
    

def preprocess(logger: echo.Logger | None = None):
    #_: load
    X, Y, num_nodes, edge_index = load_karate_club(logger)

    #_: self loop and edge normalization
    edge_index_sl = add_self_loops(edge_index, num_nodes)
    edge_norm_coefficients = compute_symmetric_normalization(edge_index_sl, num_nodes)
    
    A_norm_space = torch.sparse_coo_tensor(edge_index_sl, edge_norm_coefficients, (num_nodes, num_nodes), device=DeviceConfig.cuda)

    return X, Y, num_nodes, edge_index_sl, A_norm_space


def main():
    #_: Echo Logger
    logger = echo.Logger(sinks=[echo.ConsoleSink()])
   
    #_: sparse tensor A_norm
    X, Y, num_nodes, edge_index_sl, A_norm_sparse = preprocess(logger)
    
    print(A_norm_sparse)
   

if __name__ == "__main__":
    main()
