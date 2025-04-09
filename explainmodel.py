import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing, global_mean_pool, SAGEConv
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import seaborn as sns

class MPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNN, self).__init__(aggr='add')  
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.lin_update = torch.nn.Linear(out_channels * 2, out_channels)

    def forward(self, x, edge_index):
        m = self.propagate(edge_index, x=x)
        x = self.lin_update(torch.cat([self.lin(x), m], dim=1))
        return F.leaky_relu(x, negative_slope=0.1)

    def message(self, x_j):
        return self.lin(x_j)

class ExplainableGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, add_self_loops=True,
                 bias=True, share_weights=False, residual=False):
        super(ExplainableGATConv, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.residual = residual
        
        # Linear transformation layers
        self.lin_l = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
            
        # Attention vector
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Residual connection
        if residual:
            total_out_channels = out_channels * heads if concat else out_channels
            self.res_fc = torch.nn.Linear(in_channels, total_out_channels, bias=False)
        else:
            self.register_parameter('res_fc', None)
            
        # Bias
        if bias and concat:
            self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def forward(self, x, edge_index):
        H, C = self.heads, self.out_channels
        
        # Prepare residual connection
        if self.residual:
            res = self.res_fc(x)
        
        # Linear transformation
        x_l = self.lin_l(x).view(-1, H, C)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)
        
        # Add self-loops
        if self.add_self_loops:
            num_nodes = x_l.size(0)
            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
            
        # Use propagate for message passing
        out = self.propagate(edge_index, x=(x_l, x_r))
        
        # Process output dimensions
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            
        # Add residual connection
        if self.residual:
            out = out + res
            
        # Add bias
        if self.bias is not None:
            out = out + self.bias
            
        return out, (edge_index, self.alpha)  # Save alpha for interpretability analysis
        
    def message(self, x_j, x_i, index, size_i):
        # Calculate attention scores
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        
        # Softmax normalization
        alpha = torch_geometric.utils.softmax(alpha, index, num_nodes=size_i)
        
        # Dropout
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
            
        self.alpha = alpha  # Save for return
        
        # Apply attention weights
        return x_j * alpha.view(-1, self.heads, 1)
    
    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.lin_l.weight, gain=gain)
        if not self.share_weights:
            torch.nn.init.xavier_normal_(self.lin_r.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.att, gain=gain)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        if self.res_fc is not None:
            torch.nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class ExplainableDumplingGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(ExplainableDumplingGNN, self).__init__()
        self.mpnn = MPNN(8, hidden_channels)
        
        # Improved GAT layer configuration
        gat_config = {
            'concat': True,
            'negative_slope': 0.2,
            'dropout': 0.1,
            'add_self_loops': True,
            'share_weights': False,
            'residual': True
        }
        
        self.gat1 = ExplainableGATConv(hidden_channels, hidden_channels, 
                                      heads=8, **gat_config)
        self.gat2 = ExplainableGATConv(hidden_channels * 8, hidden_channels, 
                                      heads=8, **gat_config)
        self.gat3 = ExplainableGATConv(hidden_channels * 8, hidden_channels, 
                                      heads=8, **gat_config)
        
        self.sage = SAGEConv(hidden_channels * 8, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 2)
        
    def forward(self, data, return_attention=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        attention_weights = []
        
        # MPNN layer
        x = self.mpnn(x, edge_index)
        x = F.relu(x)
        
        # GAT layers
        x, attn1 = self.gat1(x, edge_index)
        if return_attention:
            attention_weights.append(attn1)
        x = F.elu(x)
        
        x, attn2 = self.gat2(x, edge_index)
        if return_attention:
            attention_weights.append(attn2)
        x = F.leaky_relu(x)
        
        x, attn3 = self.gat3(x, edge_index)
        if return_attention:
            attention_weights.append(attn3)
        x = F.elu(x)
        
        # SAGE layer
        x = self.sage(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.out(x)
        x = F.log_softmax(x, dim=1)
        
        return (x, attention_weights) if return_attention else x

class NoSAGEDumplingGNN(torch.nn.Module):
    """DumplingGNN variant without SAGE layer, for ablation studies"""
    def __init__(self, hidden_channels):
        super(NoSAGEDumplingGNN, self).__init__()
        self.mpnn = MPNN(8, hidden_channels)
        
        # GAT layer configuration
        gat_config = {
            'concat': True,
            'negative_slope': 0.2,
            'dropout': 0.1,
            'add_self_loops': True,
            'share_weights': False,
            'residual': True
        }
        
        self.gat1 = ExplainableGATConv(hidden_channels, hidden_channels, 
                                      heads=8, **gat_config)
        self.gat2 = ExplainableGATConv(hidden_channels * 8, hidden_channels, 
                                      heads=8, **gat_config)
        self.gat3 = ExplainableGATConv(hidden_channels * 8, hidden_channels, 
                                      heads=8, **gat_config)
        
        # Direct use of fully connected layer instead of SAGE
        self.final = torch.nn.Linear(hidden_channels * 8, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 2)
        
    def forward(self, data, return_attention=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        attention_weights = []
        
        # MPNN layer
        x = self.mpnn(x, edge_index)
        x = F.relu(x)
        
        # GAT layers
        x, attn1 = self.gat1(x, edge_index)
        if return_attention:
            attention_weights.append(attn1)
        x = F.elu(x)
        
        x, attn2 = self.gat2(x, edge_index)
        if return_attention:
            attention_weights.append(attn2)
        x = F.leaky_relu(x)
        
        x, attn3 = self.gat3(x, edge_index)
        if return_attention:
            attention_weights.append(attn3)
        x = F.elu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = self.final(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.log_softmax(x, dim=1)
        
        return (x, attention_weights) if return_attention else x

class ExplainableDumplingGNNExplainer:
    def __init__(self, model):
        self.model = model
        
    def get_attention_weights(self, data):
        """Extract attention weights from the model for a given data input"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data, return_attention=True)
            _, attention_weights = outputs
        return attention_weights
        
    def visualize_attention(self, mol, attention_weights, layer_idx=0):
        """Visualize attention weights on molecular structure
        
        Args:
            mol: RDKit molecule object
            attention_weights: List of attention weight tuples from the model
            layer_idx: Index of the GAT layer to visualize (0, 1, or 2)
        """
        edge_index, alpha = attention_weights[layer_idx]
        
        # Convert to numpy for easier processing
        edge_index = edge_index.cpu().numpy()
        alpha = alpha.cpu().numpy()
        
        # Calculate atom-level attention scores
        atom_attention = np.zeros(mol.GetNumAtoms())
        edge_count = np.zeros(mol.GetNumAtoms())
        
        # Aggregate attention scores for each atom
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < mol.GetNumAtoms() and dst < mol.GetNumAtoms():
                atom_attention[dst] += alpha[i].mean()
                edge_count[dst] += 1
        
        # Normalize by the number of edges
        for i in range(mol.GetNumAtoms()):
            if edge_count[i] > 0:
                atom_attention[i] /= edge_count[i]
        
        # Normalize to [0, 1] for visualization
        if atom_attention.max() > atom_attention.min():
            atom_attention = (atom_attention - atom_attention.min()) / (atom_attention.max() - atom_attention.min())
        
        # Create atom highlight dictionary for visualization
        atom_colors = {}
        for i, score in enumerate(atom_attention):
            # Convert attention score to RGB color (from blue to red)
            r = int(255 * score)
            g = 0
            b = int(255 * (1 - score))
            atom_colors[i] = (r, g, b)
        
        # Generate molecule image with atom highlights
        img = Draw.MolToImage(mol, highlightAtoms=list(range(mol.GetNumAtoms())), 
                              highlightAtomColors=atom_colors, size=(400, 400))
        
        # Plot both the molecule and attention distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Molecule with attention highlights
        ax1.imshow(img)
        ax1.set_title(f'GAT Layer {layer_idx+1} Attention')
        ax1.axis('off')
        
        # Attention distribution histogram
        sns.histplot(atom_attention, kde=True, ax=ax2)
        ax2.set_title('Atom Attention Distribution')
        ax2.set_xlabel('Attention Score')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        return fig, atom_attention
        
    def analyze_attention_patterns(self, data_loader):
        """Analyze attention patterns across a dataset"""
        layer_attention_stats = []
        
        for data in data_loader:
            attention_weights = self.get_attention_weights(data)
            
            # Collect statistics for each layer
            for layer_idx, (edge_index, alpha) in enumerate(attention_weights):
                alpha_np = alpha.cpu().numpy().mean(axis=1)  # Average over heads
                
                if len(layer_attention_stats) <= layer_idx:
                    layer_attention_stats.append([])
                
                layer_attention_stats[layer_idx].extend(alpha_np)
        
        # Generate summary statistics and visualizations
        for layer_idx, alpha_values in enumerate(layer_attention_stats):
            plt.figure(figsize=(8, 5))
            sns.histplot(alpha_values, kde=True)
            plt.title(f'GAT Layer {layer_idx+1} Attention Distribution')
            plt.xlabel('Attention Value')
            plt.ylabel('Frequency')
            plt.show()
            
            print(f"Layer {layer_idx+1} Attention Stats:")
            print(f"  Mean: {np.mean(alpha_values):.4f}")
            print(f"  Std: {np.std(alpha_values):.4f}")
            print(f"  Min: {np.min(alpha_values):.4f}")
            print(f"  Max: {np.max(alpha_values):.4f}")
            print()
        
        return layer_attention_stats
            
    def identify_substructures(self, mol, attention_weights, threshold=0.5):
        """Identify important substructures based on attention weights"""
        _, atom_attention = self.visualize_attention(mol, attention_weights, layer_idx=2)
        
        # Find atoms with high attention scores
        important_atoms = [i for i, score in enumerate(atom_attention) if score > threshold]
        
        # Extract substructures containing these atoms
        substructures = []
        for atom_idx in important_atoms:
            # Create a substructure centered on this atom (with neighbors)
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, 1, atom_idx)
            submol = Chem.PathToSubmol(mol, env, atomMap={})
            if submol.GetNumAtoms() > 0:
                substructures.append((atom_idx, submol))
        
        # Visualize these substructures
        if substructures:
            fig, axes = plt.subplots(1, len(substructures), figsize=(4*len(substructures), 4))
            if len(substructures) == 1:
                axes = [axes]
            
            for i, (atom_idx, submol) in enumerate(substructures):
                img = Draw.MolToImage(submol, size=(300, 300))
                axes[i].imshow(img)
                axes[i].set_title(f'Centered on Atom {atom_idx}')
                axes[i].axis('off')
            
            plt.tight_layout()
            return fig, substructures
        
        return None, []
        
    def analyze_case(self, data, mol):
        """Comprehensive analysis of a single molecule"""
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            output, attention_weights = self.model(data, return_attention=True)
            pred_class = output.argmax(dim=1).item()
            pred_prob = torch.exp(output)[0, pred_class].item()
        
        print(f"Prediction: Class {pred_class} with probability {pred_prob:.4f}")
        
        # Visualize attention for each layer
        figs = []
        for layer_idx in range(len(attention_weights)):
            fig, _ = self.visualize_attention(mol, attention_weights, layer_idx)
            figs.append(fig)
            plt.show()
        
        # Identify important substructures
        substructure_fig, substructures = self.identify_substructures(mol, attention_weights)
        if substructure_fig:
            plt.show()
        
        # Analyze attention statistics
        for layer_idx, (edge_index, alpha) in enumerate(attention_weights):
            alpha_np = alpha.cpu().numpy()
            print(f"Layer {layer_idx+1} Attention Stats:")
            print(f"  Mean: {alpha_np.mean():.4f}")
            print(f"  Std: {alpha_np.std():.4f}")
            print(f"  Min: {alpha_np.min():.4f}")
            print(f"  Max: {alpha_np.max():.4f}")
        
        return {
            'prediction': (pred_class, pred_prob),
            'attention_figs': figs,
            'substructure_fig': substructure_fig,
            'substructures': substructures,
            'attention_weights': attention_weights
        }