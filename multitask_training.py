import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Optional

from dataset import SceneGraphDataset
from models import create_training_model


class MultiTaskMaskedTrainer:
    """
    Multi-task masked trainer that implements the exact training plan you described.
    Alternates between Task A (Masked Node Prediction) and Task B (Masked Edge Prediction).
    """
    
    def __init__(self, 
                 model,
                 dataset: SceneGraphDataset,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 node_mask_ratio: float = 0.15,
                 edge_mask_ratio: float = 0.15,
                 task_switch_probability: float = 0.5):
        """
        Initialize the multi-task masked trainer.
        
        Args:
            model: Training model with encoder and prediction heads
            dataset: Scene graph dataset
            device: Device to train on
            node_mask_ratio: Fraction of nodes to mask (15% as suggested)
            edge_mask_ratio: Fraction of edges to mask (15% as suggested)
            task_switch_probability: Probability of choosing Task A vs Task B
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.node_mask_ratio = node_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio
        self.task_switch_probability = task_switch_probability
        
        # Loss functions as specified
        self.mse_loss = nn.MSELoss()  # For bbox coordinate regression
        self.cross_entropy_loss = nn.CrossEntropyLoss()  # For relation prediction
        
        # AdamW optimizer with parameters from all three models (encoder + both heads)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=10, verbose=True
        )
        
        # Training statistics
        self.task_a_count = 0
        self.task_b_count = 0
        
    def mask_nodes_task_a(self, data):
        """
        Task A: Masked Node Prediction
        Select 15% of nodes, store their bbox coordinates (center + extent), replace with zeros.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            tuple: (masked_data, masked_node_indices, original_bbox_coordinates)
        """
        num_nodes = data.x.size(0)
        num_masked = max(1, int(num_nodes * self.node_mask_ratio))  # At least 1 node
        
        # Random selection of nodes to mask
        masked_indices = torch.randperm(num_nodes)[:num_masked]
        
        # Store original bbox coordinates (positions 0-5: center[3] + extent[3])
        original_bbox = data.x[masked_indices, 0:6].clone()
        
        # Create masked data
        masked_data = data.clone()
        # Zero out bbox coordinates for masked nodes
        masked_data.x[masked_indices, 0:6] = 0
        
        return masked_data, masked_indices, original_bbox
    
    def mask_edges_task_b(self, data):
        """
        Task B: Masked Relation Prediction
        Select 15% of edges, store their relation IDs.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            tuple: (data, masked_edge_indices, original_relation_ids, masked_edge_pairs)
        """
        num_edges = data.edge_index.size(1)
        
        if num_edges == 0:
            return data, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(2, 0, dtype=torch.long)
        
        num_masked = max(1, int(num_edges * self.edge_mask_ratio))  # At least 1 edge
        
        # Random selection of edges to mask
        masked_edge_indices = torch.randperm(num_edges)[:num_masked]
        
        # Store original relation IDs
        original_relation_ids = data.edge_attr[masked_edge_indices].clone()
        
        # Store the actual edge pairs (source, target) for prediction
        masked_edge_pairs = data.edge_index[:, masked_edge_indices].clone()
        
        return data, masked_edge_indices, original_relation_ids, masked_edge_pairs
    
    def train_step(self, batch):
        """
        Single training step implementing the multi-task masked training plan.
        
        Args:
            batch: Batch of PyTorch Geometric Data objects
            
        Returns:
            dict: Dictionary of losses and task information
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Choose Task: if torch.rand(1).item() < 0.5: go to Task A, otherwise go to Task B
        use_task_a = torch.rand(1).item() < self.task_switch_probability
        
        if use_task_a:
            return self._train_task_a(batch)
        else:
            return self._train_task_b(batch)
    
    def _train_task_a(self, batch):
        """
        Task A: Masked Node Prediction
        """
        self.task_a_count += 1
        
        total_loss = 0
        num_graphs = batch.batch.max().item() + 1 if batch.batch is not None else 1
        num_processed = 0
        
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
            
            # Masking: Select subset of nodes, store original bbox coordinates
            masked_data, masked_indices, original_bbox = self.mask_nodes_task_a(single_data)
            
            if len(masked_indices) == 0:
                continue
            
            # Forward Pass: Get final_node_embeddings from GNN encoder
            outputs = self.model(masked_data, return_predictions=False)
            final_node_embeddings = outputs['node_embeddings']
            
            # Prediction: Pass embeddings of masked nodes through NodePredictionHead
            masked_node_embeddings = final_node_embeddings[masked_indices]
            predicted_bbox = self.model.node_prediction_head(masked_node_embeddings)
            
            # Loss: MSELoss(predicted_bbox, original_bbox)
            loss = self.mse_loss(predicted_bbox, original_bbox)
            
            total_loss += loss
            num_processed += 1
        
        # Average loss across graphs
        if num_processed > 0:
            total_loss = total_loss / num_processed
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return {
            'task': 'A',
            'loss': total_loss.item() if num_processed > 0 else 0,
            'num_processed': num_processed
        }
    
    def _train_task_b(self, batch):
        """
        Task B: Masked Relation Prediction
        """
        self.task_b_count += 1
        
        total_loss = 0
        num_graphs = batch.batch.max().item() + 1 if batch.batch is not None else 1
        num_processed = 0
        
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
            
            # Masking: Select subset of edges, store original relation IDs
            data, masked_edge_indices, original_relation_ids, masked_edge_pairs = self.mask_edges_task_b(single_data)
            
            if len(masked_edge_indices) == 0:
                continue
            
            # Forward Pass: Get final_node_embeddings from GNN encoder
            outputs = self.model(data, return_predictions=False)
            final_node_embeddings = outputs['node_embeddings']
            
            # Prediction: For each masked edge, get source (h_i) and target (h_j) embeddings
            source_embeddings = final_node_embeddings[masked_edge_pairs[0]]  # h_i
            target_embeddings = final_node_embeddings[masked_edge_pairs[1]]  # h_j
            
            # Concatenate them and pass through EdgePredictionHead
            concatenated_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
            predicted_relation_logits = self.model.edge_prediction_head(concatenated_embeddings)
            
            # Loss: CrossEntropyLoss(predicted_relation_logits, original_relation_ids)
            loss = self.cross_entropy_loss(predicted_relation_logits, original_relation_ids)
            
            total_loss += loss
            num_processed += 1
        
        # Average loss across graphs
        if num_processed > 0:
            total_loss = total_loss / num_processed
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return {
            'task': 'B',
            'loss': total_loss.item() if num_processed > 0 else 0,
            'num_processed': num_processed
        }
    
    def train(self, 
              num_epochs: int = 100,
              batch_size: int = 8,
              save_path: str = 'multitask_masked_model.pth',
              log_every: int = 10):
        """
        Train the model with multi-task masked prediction.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            save_path: Path to save the best model
            log_every: Log statistics every N epochs
        """
        # Create dataloader
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        task_a_losses = []
        task_b_losses = []
        
        print(f"Starting multi-task masked training for {num_epochs} epochs")
        print(f"Task A: Masked Node Prediction (bbox coordinate reconstruction)")
        print(f"Task B: Masked Edge Prediction (Relation prediction)")
        print(f"Task switch probability: {self.task_switch_probability}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            epoch_task_a_losses = []
            epoch_task_b_losses = []
            
            # Training loop
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch in progress_bar:
                batch = batch.to(self.device)
                result = self.train_step(batch)
                
                if result['task'] == 'A':
                    epoch_task_a_losses.append(result['loss'])
                else:
                    epoch_task_b_losses.append(result['loss'])
                
                # Update progress bar
                recent_task = result['task']
                recent_loss = result['loss']
                progress_bar.set_postfix({
                    'Task': recent_task,
                    'Loss': f'{recent_loss:.4f}'
                })
            
            # Calculate average losses
            avg_task_a_loss = np.mean(epoch_task_a_losses) if epoch_task_a_losses else 0
            avg_task_b_loss = np.mean(epoch_task_b_losses) if epoch_task_b_losses else 0
            combined_loss = (avg_task_a_loss + avg_task_b_loss) / 2
            
            task_a_losses.append(avg_task_a_loss)
            task_b_losses.append(avg_task_b_loss)
            
            # Logging
            if (epoch + 1) % log_every == 0:
                print(f'\nEpoch {epoch+1}/{num_epochs}:')
                print(f"  Task A (Node) Loss: {avg_task_a_loss:.4f} [{len(epoch_task_a_losses)} batches]")
                print(f"  Task B (Edge) Loss: {avg_task_b_loss:.4f} [{len(epoch_task_b_losses)} batches]")
                print(f"  Combined Loss: {combined_loss:.4f}")
                print(f"  Task A total runs: {self.task_a_count}")
                print(f"  Task B total runs: {self.task_b_count}")
            
            # Learning rate scheduling
            self.scheduler.step(combined_loss)
            
            # Save best model
            if combined_loss < best_loss and combined_loss > 0:
                best_loss = combined_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': best_loss,
                    'task_a_losses': task_a_losses,
                    'task_b_losses': task_b_losses,
                    'relation_mapping': self.dataset.get_relation_mapping(),
                    'task_counts': {
                        'task_a': self.task_a_count,
                        'task_b': self.task_b_count
                    }
                }, save_path)
                
                if (epoch + 1) % log_every == 0:
                    print(f"  ✓ Saved new best model with combined loss {best_loss:.4f}")
        
        print(f"\nTraining completed!")
        print(f"Final statistics:")
        print(f"  Task A executed: {self.task_a_count} times")
        print(f"  Task B executed: {self.task_b_count} times")
        print(f"  Best combined loss: {best_loss:.4f}")
        print(f"  Model saved to: {save_path}")


def main():
    """
    Main training script implementing the exact multi-task masked training plan.
    """
    print("Multi-Task Masked Training for Scene Graph Encoder")
    print("=" * 60)
    
    # Load dataset
    data_path = "./data"
    dataset = SceneGraphDataset(data_path)
    
    if len(dataset) == 0:
        print("No data found. Creating sample data for demonstration...")
        from data_utils import create_sample_data
        create_sample_data(data_path, num_samples=30)
        dataset = SceneGraphDataset(data_path)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Number of relations: {dataset.get_num_relations()}")
    print(f"Relations: {list(dataset.get_relation_mapping().keys())}")
    
    # Create training model with all components
    model = create_training_model(
        dataset,
        node_embedding_dim=128,
        edge_embedding_dim=32,
        bbox_dim=6
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create multi-task trainer
    trainer = MultiTaskMaskedTrainer(
        model=model,
        dataset=dataset,
        node_mask_ratio=0.15,        # 15% as specified
        edge_mask_ratio=0.15,        # 15% as specified  
        task_switch_probability=0.5   # 50% chance for each task
    )
    
    # Train with the exact plan you described
    trainer.train(
        num_epochs=100,
        batch_size=6,
        save_path='multitask_scene_encoder.pth',
        log_every=5
    )
    
    print("\n" + "=" * 60)
    print("Training completed! Next steps:")
    print("1. The encoder is now pre-trained with both tasks")
    print("2. Use the trained encoder for downstream scene embedding tasks")
    print("3. Load weights into inference model for [1, 32] scene embeddings")
    
    # Demonstrate how to use the trained model for inference
    print("\nDemonstrating inference with trained model...")
    
    from models import create_inference_model
    
    # Create inference model and load trained encoder weights
    inference_model = create_inference_model(dataset, scene_embedding_dim=32)
    inference_model.load_encoder_weights(model)
    
    # Test scene embedding generation
    sample_data = dataset[0]
    inference_model.eval()
    with torch.no_grad():
        scene_embedding = inference_model(sample_data)
        print(f"✓ Generated scene embedding: {scene_embedding.shape}")  # [1, 32]
        print(f"✓ Ready for downstream tasks!")


if __name__ == "__main__":
    main()
