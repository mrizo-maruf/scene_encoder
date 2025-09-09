import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Optional

from dataset import SceneGraphDataset
from models import GNNSceneEmbeddingNetwork_Training, GNNSceneEmbeddingNetwork_Inference


class MaskedTrainer:
    """
    Trainer for masked node and edge prediction tasks.
    """
    
    def __init__(self, 
                 model: GNNSceneEmbeddingNetwork_Training,
                 dataset: SceneGraphDataset,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 node_mask_ratio: float = 0.15,
                 edge_mask_ratio: float = 0.15):
        """
        Initialize the masked trainer.
        
        Args:
            model: Training model with encoder and prediction heads
            dataset: Scene graph dataset
            device: Device to train on
            node_mask_ratio: Fraction of nodes to mask
            edge_mask_ratio: Fraction of edges to mask
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.node_mask_ratio = node_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio
        
        # Loss functions
        self.node_criterion = nn.MSELoss()
        self.edge_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
    def mask_nodes(self, data):
        """
        Mask random nodes by replacing their CLIP descriptors with zeros.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            tuple: (masked_data, mask, original_clip_features)
        """
        num_nodes = data.x.size(0)
        num_masked = int(num_nodes * self.node_mask_ratio)
        
        if num_masked == 0:
            # No masking for very small graphs
            return data, torch.zeros(num_nodes, dtype=torch.bool), data.x[:, 6:518]
        
        # Random selection of nodes to mask
        masked_indices = torch.randperm(num_nodes)[:num_masked]
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[masked_indices] = True
        
        # Store original CLIP features (positions 6-517, 512 dimensions)
        original_clip = data.x[:, 6:518].clone()
        
        # Create masked data
        masked_data = data.clone()
        # Zero out CLIP descriptors for masked nodes
        masked_data.x[masked_indices, 6:518] = 0
        
        return masked_data, mask, original_clip
    
    def mask_edges(self, data):
        """
        Mask random edges by removing them from the graph.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            tuple: (masked_data, edge_mask, original_edge_attr)
        """
        num_edges = data.edge_index.size(1)
        num_masked = int(num_edges * self.edge_mask_ratio)
        
        if num_masked == 0 or num_edges == 0:
            # No masking for graphs with no/few edges
            return data, torch.zeros(num_edges, dtype=torch.bool), data.edge_attr
        
        # Random selection of edges to mask
        masked_edge_indices = torch.randperm(num_edges)[:num_masked]
        edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        edge_mask[masked_edge_indices] = True
        
        # Store original edge attributes
        original_edge_attr = data.edge_attr.clone()
        
        # Create masked data by removing edges
        masked_data = data.clone()
        remaining_indices = ~edge_mask
        masked_data.edge_index = data.edge_index[:, remaining_indices]
        masked_data.edge_attr = data.edge_attr[remaining_indices]
        
        return masked_data, edge_mask, original_edge_attr
    
    def create_negative_edges(self, data, num_negatives_per_positive: int = 1):
        """
        Create negative edge samples for edge prediction task.
        
        Args:
            data: PyTorch Geometric Data object
            num_negatives_per_positive: Number of negative samples per positive edge
            
        Returns:
            tuple: (negative_edge_index, negative_labels)
        """
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)
        num_negatives = num_edges * num_negatives_per_positive
        
        if num_nodes < 2:
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        # Convert edge_index to set for fast lookup
        existing_edges = set()
        for i in range(num_edges):
            src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            existing_edges.add((src, dst))
        
        # Sample negative edges
        negative_edges = []
        attempts = 0
        max_attempts = num_negatives * 10  # Avoid infinite loop
        
        while len(negative_edges) < num_negatives and attempts < max_attempts:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            
            if src != dst and (src, dst) not in existing_edges:
                negative_edges.append([src, dst])
            
            attempts += 1
        
        if len(negative_edges) == 0:
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        negative_edge_index = torch.tensor(negative_edges, dtype=torch.long).t()
        # Use a special "no relation" class (assume it's the last class)
        negative_labels = torch.full((len(negative_edges),), self.dataset.get_num_relations() - 1, dtype=torch.long)
        
        return negative_edge_index, negative_labels
    
    def train_step(self, batch):
        """
        Single training step with masked prediction.
        
        Args:
            batch: Batch of PyTorch Geometric Data objects
            
        Returns:
            dict: Dictionary of losses
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        total_node_loss = 0
        total_edge_loss = 0
        num_graphs = batch.batch.max().item() + 1 if batch.batch is not None else 1
        
        # Process each graph in the batch separately for masking
        graph_start_idx = 0
        for graph_idx in range(num_graphs):
            # Extract single graph from batch
            if batch.batch is not None:
                graph_mask = batch.batch == graph_idx
                graph_end_idx = graph_start_idx + graph_mask.sum().item()
                
                # Extract graph data
                graph_x = batch.x[graph_mask]
                graph_edge_mask = (batch.batch[batch.edge_index[0]] == graph_idx) & \
                                 (batch.batch[batch.edge_index[1]] == graph_idx)
                graph_edge_index = batch.edge_index[:, graph_edge_mask] - graph_start_idx
                graph_edge_attr = batch.edge_attr[graph_edge_mask]
                
                # Create single graph data object
                from torch_geometric.data import Data
                single_data = Data(x=graph_x, edge_index=graph_edge_index, edge_attr=graph_edge_attr)
                
                graph_start_idx = graph_end_idx
            else:
                single_data = batch
            
            # Node masking
            masked_data_nodes, node_mask, original_clip = self.mask_nodes(single_data)
            
            if node_mask.any():
                # Forward pass on node-masked data
                outputs = self.model(masked_data_nodes, return_predictions=True)
                node_predictions = outputs['node_predictions']
                
                # Node reconstruction loss (only on masked nodes)
                masked_node_predictions = node_predictions[node_mask]
                masked_original_clip = original_clip[node_mask]
                
                if masked_node_predictions.size(0) > 0:
                    node_loss = self.node_criterion(masked_node_predictions, masked_original_clip)
                    total_node_loss += node_loss
            
            # Edge masking
            masked_data_edges, edge_mask, original_edge_attr = self.mask_edges(single_data)
            
            if edge_mask.any():
                # Forward pass on edge-masked data
                outputs = self.model(masked_data_edges, return_predictions=True)
                node_embeddings = outputs['node_embeddings']
                
                # Get masked edges for prediction
                masked_edge_indices = torch.where(edge_mask)[0]
                masked_edges = single_data.edge_index[:, masked_edge_indices]
                masked_edge_labels = original_edge_attr[masked_edge_indices]
                
                # Create negative edges
                neg_edge_index, neg_labels = self.create_negative_edges(single_data)
                
                # Combine positive (masked) and negative edges
                if neg_edge_index.size(1) > 0:
                    all_edge_index = torch.cat([masked_edges, neg_edge_index], dim=1)
                    all_edge_labels = torch.cat([masked_edge_labels, neg_labels])
                else:
                    all_edge_index = masked_edges
                    all_edge_labels = masked_edge_labels
                
                if all_edge_index.size(1) > 0:
                    # Get source and target embeddings
                    source_embeddings = node_embeddings[all_edge_index[0]]
                    target_embeddings = node_embeddings[all_edge_index[1]]
                    concatenated_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
                    
                    # Edge prediction
                    edge_predictions = self.model.edge_prediction_head(concatenated_embeddings)
                    edge_loss = self.edge_criterion(edge_predictions, all_edge_labels)
                    total_edge_loss += edge_loss
        
        # Combine losses
        total_loss = total_node_loss + total_edge_loss
        
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return {
            'total_loss': total_loss.item() if total_loss > 0 else 0,
            'node_loss': total_node_loss.item() if isinstance(total_node_loss, torch.Tensor) else 0,
            'edge_loss': total_edge_loss.item() if isinstance(total_edge_loss, torch.Tensor) else 0
        }
    
    def train(self, 
              num_epochs: int = 100,
              batch_size: int = 8,
              validate_every: int = 10,
              save_path: str = 'masked_model.pth'):
        """
        Train the model with masked prediction tasks.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validate_every: Validate every N epochs
            save_path: Path to save the best model
        """
        # Create dataloader
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_losses = {'total_loss': 0, 'node_loss': 0, 'edge_loss': 0}
            num_batches = 0
            
            # Training loop
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch in progress_bar:
                batch = batch.to(self.device)
                losses = self.train_step(batch)
                
                for key in epoch_losses:
                    epoch_losses[key] += losses[key]
                num_batches += 1
                
                # Update progress bar
                avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
                progress_bar.set_postfix(avg_losses)
            
            # Average losses
            avg_losses = {k: v/num_batches if num_batches > 0 else 0 for k, v in epoch_losses.items()}
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f"  Total Loss: {avg_losses['total_loss']:.4f}")
            print(f"  Node Loss: {avg_losses['node_loss']:.4f}")
            print(f"  Edge Loss: {avg_losses['edge_loss']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(avg_losses['total_loss'])
            
            # Save best model
            if avg_losses['total_loss'] < best_loss:
                best_loss = avg_losses['total_loss']
                # Convert relation mapping to regular dict to avoid pickle issues
                relation_mapping = self.dataset.get_relation_mapping()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                    'relation_mapping': relation_mapping
                }, save_path)
                print(f"  Saved new best model with loss {best_loss:.4f}")
            
            print()


def main():
    """
    Example training script.
    """
    # Load dataset
    data_path = "./data"
    dataset = SceneGraphDataset(data_path, relations_txt="/home/rizo/mipt_ccm/scene_encoder/data/vlsat_relations.txt")
    
    if len(dataset) == 0:
        print("No data found. Creating sample data for demonstration...")
        from data_utils import create_sample_data
        create_sample_data(data_path, num_samples=20)
        dataset = SceneGraphDataset(data_path)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Number of relations: {dataset.get_num_relations()}")
    
    # Create training model
    from models import create_training_model
    model = create_training_model(dataset)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = MaskedTrainer(
        model=model,
        dataset=dataset,
        node_mask_ratio=0.15,
        edge_mask_ratio=0.15
    )
    
    # Train
    trainer.train(
        num_epochs=50,
        batch_size=4,
        validate_every=10,
        save_path='masked_scene_encoder.pth'
    )
    
    print("Training completed!")
    
    # Example: Create inference model and load trained weights
    print("\nCreating inference model...")
    from models import create_inference_model
    inference_model = create_inference_model(dataset, scene_embedding_dim=32)
    inference_model.load_encoder_weights(model)
    
    # Test inference
    sample_data = dataset[0]
    with torch.no_grad():
        scene_embedding = inference_model(sample_data)
        print(f"Scene embedding shape: {scene_embedding.shape}")  # Should be [1, 32]


if __name__ == "__main__":
    main()
