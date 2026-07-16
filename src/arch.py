import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, activation=None, bias:bool = True) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation = activation

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.b = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('b', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        
        if self.bias:
            nn.init.zeros_(self.b)

    def forward(self, X, A_norm):
        """
            have to multiply in the same order: A_norm (num_nodes, num_nodes),  X (num_nodes, in_features),  W (in_features x out_features)
            
            sparse, dense, dense

            X with W first, then A with result, since num_nodes is big compared to feature vector dimensions, faster on CUDA
        """

        intermediate = X @ self.W
        out = torch.sparse.mm(A_norm, intermediate)
        if self.bias:
            out = out + self.b
        if self.activation:
            out = self.activation(out)

        return out



class KarateClubGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout_rate=0.5):
        super().__init__()

        self.conv1 = GraphConvolution(
            in_features=num_features,
            out_features=hidden_dim,
            activation=nn.ReLU()
        )

        self.conv2 = GraphConvolution(
            in_features=hidden_dim,
            out_features=num_classes,
            activation=None
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X, A_norm):
        h = self.conv1(X, A_norm)
        h = self.dropout(h)
        logits = self.conv2(h, A_norm)

        return logits
