import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from dataset import SceneGraphDataset, create_dataloader
from models import (
    GNNSceneEmbeddingNetwork_Training, 
    GNNSceneEmbeddingNetwork_Inference,
    create_training_model,
    create_inference_model
)


class SceneGraphGNN(nn.Module):
    """
    Simple GNN model for scene graph processing (backward compatibility).
    """
    
    def __init__(self, node_input_dim: int = 518, hidden_dim: int = 128, 
                 num_relations: int = 10, output_dim: int = 64):
        """
        Initialize the GNN model.
        
        Args:
            node_input_dim (int): Input dimension for node features (518 for your case)
            hidden_dim (int): Hidden dimension
            num_relations (int): Number of unique relations for edge embeddings
            output_dim (int): Output dimension for graph-level representation
        """
        super().__init__()
        
        # Node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge relation embeddings
        self.edge_embedding = nn.Embedding(num_relations, hidden_dim // 4)
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Graph-level output
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, 518]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_attr (torch.Tensor): Edge attributes [num_edges]
            batch (torch.Tensor): Batch assignment for each node
            
        Returns:
            torch.Tensor: Graph-level representation
        """
        # Encode node features
        x = self.node_encoder(x)
        
        # Get edge embeddings (though GCNConv doesn't use them directly)
        # This is here for potential future use with edge-aware GNN layers
        edge_emb = self.edge_embedding(edge_attr)
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # Graph-level pooling
        if batch is None:
            # Single graph
            graph_repr = torch.mean(x, dim=0, keepdim=True)
        else:
            # Batch of graphs
            graph_repr = global_mean_pool(x, batch)
        
        # Final output
        output = self.graph_head(graph_repr)
        
        return output


def train_example(data_path: str, epochs: int = 10):
    """
    Example training loop using the new modular architecture.
    
    Args:
        data_path (str): Path to the directory containing .pkl files
        epochs (int): Number of training epochs
    """
    # Create dataset and dataloader
    dataset = SceneGraphDataset(data_path, relations_txt="/home/rizo/mipt_ccm/scene_encoder/data/vlsat_relations.txt")
    
    if len(dataset) == 0:
        print("No data found. Please add .pkl files to the data directory.")
        return
    
    print(f"Dataset loaded: {len(dataset)} samples with {dataset.get_num_relations()} relations")
    
    # Create models using new architecture
    print("\n=== Training with New Architecture ===")
    
    # 1. Create training model for masked pretraining
    training_model = create_training_model(dataset, node_embedding_dim=128)
    print(f"Training model created with {sum(p.numel() for p in training_model.parameters())} parameters")
    
    # 2. Create inference model for scene embeddings
    inference_model = create_inference_model(dataset, scene_embedding_dim=32)
    print(f"Inference model created with {sum(p.numel() for p in inference_model.parameters())} parameters")
    
    # 3. Demonstrate training workflow
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
    
    # Example forward pass with training model
    batch = next(iter(dataloader))
    print(f"\nBatch test:")
    print(f"  Batch size: {batch.batch.max() + 1}")
    print(f"  Total nodes in batch: {batch.x.shape[0]}")
    print(f"  Total edges in batch: {batch.edge_index.shape[1]}")
    
    # Training model forward (returns node embeddings + predictions)
    training_model.eval()
    with torch.no_grad():
        training_outputs = training_model(batch)
        print(f"\nTraining model outputs:")
        print(f"  Node embeddings shape: {training_outputs['node_embeddings'].shape}")
        if training_outputs['node_predictions'] is not None:
            print(f"  Node predictions shape: {training_outputs['node_predictions'].shape}")
        if training_outputs['edge_predictions'] is not None:
            print(f"  Edge predictions shape: {training_outputs['edge_predictions'].shape}")
    
    # Inference model forward (returns scene embeddings)
    inference_model.eval()
    with torch.no_grad():
        scene_embeddings = inference_model(batch)
        print(f"\nInference model output:")
        print(f"  Scene embeddings shape: {scene_embeddings.shape}")  # Should be [batch_size, 32]
    
    # 4. Show how to transfer weights
    print(f"\n=== Weight Transfer Demo ===")
    inference_model.load_encoder_weights(training_model)
    print("Transferred encoder weights from training to inference model")
    
    # Test single graph inference
    single_graph = dataset[0]
    with torch.no_grad():
        single_scene_embedding = inference_model(single_graph)
        print(f"Single graph scene embedding shape: {single_scene_embedding.shape}")  # Should be [1, 32]
    
    # 5. Backward compatibility test
    print(f"\n=== Backward Compatibility Test ===")
    
    # Original model interface still works
    from models import GNNSceneEmbeddingNetwork_LearnedEdgeVector
    legacy_model = GNNSceneEmbeddingNetwork_LearnedEdgeVector(
        object_feature_dim=518,
        num_relations=dataset.get_num_relations(),
        node_embedding_dim=128,
        edge_embedding_dim=32,
        scene_embedding_dim=32
    )
    
    with torch.no_grad():
        legacy_output = legacy_model(single_graph)
        print(f"Legacy model output shape: {legacy_output.shape}")  # Should be [1, 32]
    
    print("\n=== All tests completed successfully! ===")
    print("\nNext steps:")
    print("1. Use masked_training.py for self-supervised pretraining")
    print("2. Use the trained encoder for downstream tasks")
    print("3. Fine-tune the inference model for specific applications")


def analyze_dataset(data_path: str):
    """
    Analyze the dataset to understand the data distribution.
    
    Args:
        data_path (str): Path to the directory containing .pkl files
    """
    dataset = SceneGraphDataset(data_path, relations_txt="/home/rizo/mipt_ccm/scene_encoder/data/vlsat_relations.txt")

    if len(dataset) == 0:
        print("No data found.")
        return
    
    print(f"Dataset Analysis:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Unique relations: {dataset.get_num_relations()}")
    print(f"  Relations: {list(dataset.get_relation_mapping())}")
    
    # Analyze graph sizes
    node_counts = []
    edge_counts = []
    
    for i in range(min(len(dataset), 100)):  # Analyze first 100 samples
        data = dataset[i]
        node_counts.append(data.num_nodes)
        edge_counts.append(data.edge_index.shape[1])
    
    print(f"\nGraph Statistics (first {len(node_counts)} samples):")
    print(f"  Nodes per graph - Mean: {sum(node_counts)/len(node_counts):.2f}, "
          f"Min: {min(node_counts)}, Max: {max(node_counts)}")
    print(f"  Edges per graph - Mean: {sum(edge_counts)/len(edge_counts):.2f}, "
          f"Min: {min(edge_counts)}, Max: {max(edge_counts)}")


if __name__ == "__main__":
    import os
    
    data_path = "/home/rizo/mipt_ccm/scene_encoder/data"
    
    if not os.path.exists(data_path):
        print(f"Creating data directory: {data_path}")
        os.makedirs(data_path)
        print("Please add your .pkl files to the data directory and run again.")
    else:
        # Analyze dataset
        analyze_dataset(data_path)
        
        # Run training example if data exists
        dataset = SceneGraphDataset(data_path, relations_txt="/home/rizo/mipt_ccm/scene_encoder/data/vlsat_relations.txt")
        if len(dataset) > 0:
            print("\nRunning training example...")
            train_example(data_path, epochs=3)
        else:
            print("No .pkl files found in data directory.")
            print("Please add your .pkl files and run again.")
