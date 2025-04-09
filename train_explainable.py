import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from explainmodel import ExplainableDumplingGNN
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from rdkit import Chem

def mol_to_graph_with_coords(mol):
    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_feature = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),     # Degree
            atom.GetTotalNumHs(), # Number of hydrogen atoms
            atom.GetImplicitValence(),  # Implicit valence
            atom.GetIsAromatic(),       # Is aromatic
        ] + list(mol.GetConformer().GetAtomPosition(atom.GetIdx()))  # Atom coordinates
        atom_features.append(atom_feature)

    # Extract edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        edge_indices.append([start_atom, end_atom])
        edge_indices.append([end_atom, start_atom])

    # Convert features and indices to PyTorch tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # Get label
    label = mol.GetProp("label")
    y = torch.tensor([float(label)], dtype=torch.long)  # Convert to long type

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    return data

def load_and_process_data(sdf_path, test_size=0.2, val_size=0.2):
    """Load and process the dataset"""
    print("Loading and processing data...")
    
    # Load SDF file
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    
    # Convert to graph data
    data_list = []
    for mol in tqdm(supplier, desc="Processing molecules"):
        if mol is not None:
            try:
                data = mol_to_graph_with_coords(mol)
                data_list.append(data)
            except:
                continue
    
    # First split out the test set
    train_val_data, test_data = train_test_split(
        data_list, 
        test_size=test_size, 
        random_state=42
    )
    
    # Split the remaining data into train and validation sets
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size,
        random_state=42
    )
    
    print(f"Dataset split complete:")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")
    
    return train_data, val_data, test_data

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 lr=0.001, weight_decay=5e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), 
                                         lr=lr, 
                                         weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, 
            min_lr=1e-6, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_auc = 0
        self.best_model = None
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data)
            loss = F.nll_loss(out, data.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            
        return total_loss / len(self.train_loader.dataset)
    
    def evaluate(self, loader, return_predictions=False):
        self.model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        y_score = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                loss = F.nll_loss(out, data.y)
                total_loss += loss.item() * data.num_graphs
                
                y_true.extend(data.y.cpu().numpy())
                y_pred.extend(out.max(1)[1].cpu().numpy())
                y_score.extend(torch.exp(out)[:, 1].cpu().numpy())
        
        # Calculate evaluation metrics
        metrics = {
            'loss': total_loss / len(loader.dataset),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_score)
        }
        
        if return_predictions:
            return metrics, (y_true, y_pred, y_score)
        return metrics
    
    def train(self, epochs, patience=20, save_path='saved_models'):
        os.makedirs(save_path, exist_ok=True)
        best_epoch = 0
        no_improve = 0
        
        for epoch in tqdm(range(epochs), desc='Training'):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['auc'])
            
            # Save best model
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_model = self.model.state_dict()
                torch.save(self.best_model, 
                         os.path.join(save_path, 'best_model.pth'))
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}:')
                print(f'Train Loss: {train_loss:.4f}')
                print(f'Val Loss: {val_metrics["loss"]:.4f}')
                print(f'Val AUC: {val_metrics["auc"]:.4f}')
                print(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
                print('---')
        
        print(f'\nBest validation AUC: {self.best_val_auc:.4f} at epoch {best_epoch}')
        return self.best_model
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        
        # Validation metrics
        metrics_df = pd.DataFrame(self.val_metrics)
        for col in metrics_df.columns:
            if col != 'loss':
                ax2.plot(metrics_df[col], label=col.capitalize())
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.set_title('Validation Metrics')
        
        plt.tight_layout()
        plt.show()

def main():
    # Load and process data
    sdf_path = 'train_chembl.sdf'
    print(f"Looking for SDF file at: {os.path.abspath(sdf_path)}")
    
    if not os.path.exists(sdf_path):
        print(f"Error: SDF file not found at {sdf_path}")
        return
        
    try:
        train_data, val_data, test_data = load_and_process_data(
            sdf_path,
            test_size=0.2,
            val_size=0.2
        )
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        test_loader = DataLoader(test_data, batch_size=32)
        
        # Initialize model
        hidden_channels = 32
        model = ExplainableDumplingGNN(hidden_channels)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=0.0001,
            weight_decay=1e-5
        )
        
        # Train model
        best_model = trainer.train(epochs=1000, patience=50)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate on test set
        model.load_state_dict(best_model)
        test_metrics = trainer.evaluate(test_loader)
        
        print("\nTest Set Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main() 