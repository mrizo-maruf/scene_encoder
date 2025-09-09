import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data


class GNNSceneEncoder(nn.Module):
    """
    GNN Encoder that processes scene graphs and returns node embeddings.
    This is the shared encoder used for both training and inference.
    """
    def __init__(self, 
                 object_feature_dim=518, 
                 num_relations=26,
                 node_embedding_dim=128, 
                 edge_embedding_dim=32):
        """
        Initialize the GNN Scene Encoder.
        
        Args:
            object_feature_dim (int): Dimensionality of input node features (518 for your case)
            num_relations (int): Number of unique relation types
            node_embedding_dim (int): Dimensionality of the node embeddings
            edge_embedding_dim (int): Dimensionality of the edge embeddings
        """
        super(GNNSceneEncoder, self).__init__()
        
        self.node_embedding_dim = node_embedding_dim
        
        # 1. Node Feature Encoder: Maps high-dim raw features to embedding space
        self.node_encoder = nn.Sequential(
            nn.Linear(object_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, node_embedding_dim)
        )
        
        # 2. Edge Feature Embedding: Creates dense vectors for relation types
        self.rel_embedding = nn.Embedding(num_relations, edge_embedding_dim)

        # 3. GNN Layers: Use GATConv with edge features
        self.conv1 = GATConv(
            in_channels=node_embedding_dim,
            out_channels=node_embedding_dim,
            edge_dim=edge_embedding_dim,
            heads=4,
            concat=False,  # Average attention heads
            dropout=0.1
        )
        
        self.conv2 = GATConv(
            in_channels=node_embedding_dim,
            out_channels=node_embedding_dim,
            edge_dim=edge_embedding_dim,
            heads=4,
            concat=False,
            dropout=0.1
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data):
        """
        Forward pass that returns node embeddings.
        
        Args:
            data: PyTorch Geometric Data object with x, edge_index, edge_attr
            
        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, node_embedding_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. Encode node features
        x = self.node_encoder(x)  # Shape: [num_nodes, node_embedding_dim]

        # 2. Encode edge features
        edge_attr_embedded = self.rel_embedding(edge_attr)  # Shape: [num_edges, edge_embedding_dim]

        # 3. Message Passing with GNN layers
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr_embedded))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr_embedded))
        
        return x  # Return node embeddings for training


class NodePredictionHead(nn.Module):
    """
    Prediction head for reconstructing CLIP descriptors from node embeddings.
    Used in masked node prediction task.
    """
    def __init__(self, node_embedding_dim=128, clip_dim=512):
        """
        Initialize the node prediction head.
        
        Args:
            node_embedding_dim (int): Dimensionality of input node embeddings
            clip_dim (int): Dimensionality of CLIP descriptors to predict
        """
        super(NodePredictionHead, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(node_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, clip_dim)
        )
    
    def forward(self, node_embeddings):
        """
        Predict CLIP descriptors from node embeddings.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings [num_nodes, node_embedding_dim]
            
        Returns:
            torch.Tensor: Predicted CLIP descriptors [num_nodes, clip_dim]
        """
        return self.mlp(node_embeddings)


class EdgePredictionHead(nn.Module):
    """
    Prediction head for predicting edge relations from node pairs.
    Used in masked edge prediction task.
    """
    def __init__(self, node_embedding_dim=128, num_relations=26):
        """
        Initialize the edge prediction head.
        
        Args:
            node_embedding_dim (int): Dimensionality of node embeddings
            num_relations (int): Number of relation types to predict
        """
        super(EdgePredictionHead, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_relations)
        )
    
    def forward(self, concatenated_node_embeddings):
        """
        Predict relation types from concatenated node embeddings.
        
        Args:
            concatenated_node_embeddings (torch.Tensor): Concatenated pairs [num_edges, node_embedding_dim * 2]
            
        Returns:
            torch.Tensor: Relation logits [num_edges, num_relations]
        """
        return self.mlp(concatenated_node_embeddings)


class SceneEmbeddingHead(nn.Module):
    """
    Head for generating final scene-level embeddings from node embeddings.
    Used during inference for downstream tasks.
    """
    def __init__(self, node_embedding_dim=128, scene_embedding_dim=32):
        """
        Initialize the scene embedding head.
        
        Args:
            node_embedding_dim (int): Dimensionality of input node embeddings
            scene_embedding_dim (int): Dimensionality of final scene embedding
        """
        super(SceneEmbeddingHead, self).__init__()
        
        self.readout_net = nn.Sequential(
            nn.Linear(node_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, scene_embedding_dim)
        )
    
    def forward(self, node_embeddings, batch=None):
        """
        Generate scene-level embedding from node embeddings.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings [num_nodes, node_embedding_dim]
            batch (torch.Tensor, optional): Batch assignment for each node
            
        Returns:
            torch.Tensor: Scene embedding [batch_size, scene_embedding_dim]
        """
        # Global pooling to get scene-level representation
        if batch is None:
            # Single graph case
            pooled = torch.mean(node_embeddings, dim=0, keepdim=True)
        else:
            # Batch case
            pooled = global_mean_pool(node_embeddings, batch)
        
        # Final scene embedding
        scene_embedding = self.readout_net(pooled)
        return scene_embedding


class GNNSceneEmbeddingNetwork_Training(nn.Module):
    """
    Complete model for masked training with both node and edge prediction heads.
    """
    def __init__(self, 
                 object_feature_dim=518, 
                 num_relations=26,
                 node_embedding_dim=128, 
                 edge_embedding_dim=32,
                 clip_dim=512):
        """
        Initialize the training model with encoder and prediction heads.
        
        Args:
            object_feature_dim (int): Dimensionality of input node features
            num_relations (int): Number of unique relation types
            node_embedding_dim (int): Dimensionality of node embeddings
            edge_embedding_dim (int): Dimensionality of edge embeddings
            clip_dim (int): Dimensionality of CLIP descriptors
        """
        super(GNNSceneEmbeddingNetwork_Training, self).__init__()
        
        # Shared encoder
        self.encoder = GNNSceneEncoder(
            object_feature_dim=object_feature_dim,
            num_relations=num_relations,
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim
        )
        
        # Prediction heads
        self.node_prediction_head = NodePredictionHead(
            node_embedding_dim=node_embedding_dim,
            clip_dim=clip_dim
        )
        
        self.edge_prediction_head = EdgePredictionHead(
            node_embedding_dim=node_embedding_dim,
            num_relations=num_relations
        )
    
    def forward(self, data, return_predictions=True):
        """
        Forward pass for training.
        
        Args:
            data: PyTorch Geometric Data object
            return_predictions (bool): Whether to return prediction head outputs
            
        Returns:
            dict: Dictionary containing node_embeddings and optionally predictions
        """
        # Get node embeddings from encoder
        node_embeddings = self.encoder(data)
        
        outputs = {'node_embeddings': node_embeddings}
        
        if return_predictions:
            # Node predictions (CLIP reconstruction)
            node_predictions = self.node_prediction_head(node_embeddings)
            outputs['node_predictions'] = node_predictions
            
            # Edge predictions
            if data.edge_index.size(1) > 0:  # Only if edges exist
                # Get source and target node embeddings
                source_embeddings = node_embeddings[data.edge_index[0]]
                target_embeddings = node_embeddings[data.edge_index[1]]
                
                # Concatenate for edge prediction
                concatenated_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
                edge_predictions = self.edge_prediction_head(concatenated_embeddings)
                outputs['edge_predictions'] = edge_predictions
            else:
                outputs['edge_predictions'] = None
        
        return outputs


class GNNSceneEmbeddingNetwork_Inference(nn.Module):
    """
    Complete model for inference that generates scene-level embeddings.
    """
    def __init__(self, 
                 object_feature_dim=518, 
                 num_relations=26,
                 node_embedding_dim=128, 
                 edge_embedding_dim=32,
                 scene_embedding_dim=32):
        """
        Initialize the inference model.
        
        Args:
            object_feature_dim (int): Dimensionality of input node features
            num_relations (int): Number of unique relation types
            node_embedding_dim (int): Dimensionality of node embeddings
            edge_embedding_dim (int): Dimensionality of edge embeddings
            scene_embedding_dim (int): Dimensionality of final scene embedding
        """
        super(GNNSceneEmbeddingNetwork_Inference, self).__init__()
        
        # Shared encoder
        self.encoder = GNNSceneEncoder(
            object_feature_dim=object_feature_dim,
            num_relations=num_relations,
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim
        )
        
        # Scene embedding head
        self.scene_head = SceneEmbeddingHead(
            node_embedding_dim=node_embedding_dim,
            scene_embedding_dim=scene_embedding_dim
        )
    
    def forward(self, data):
        """
        Forward pass for inference.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            torch.Tensor: Scene embedding [batch_size, scene_embedding_dim]
        """
        # Get node embeddings
        node_embeddings = self.encoder(data)
        
        # Generate scene embedding
        scene_embedding = self.scene_head(node_embeddings, data.batch)
        
        return scene_embedding
    
    def load_encoder_weights(self, training_model):
        """
        Load encoder weights from a trained model.
        
        Args:
            training_model (GNNSceneEmbeddingNetwork_Training): Trained model
        """
        self.encoder.load_state_dict(training_model.encoder.state_dict())


# Backward compatibility: Original model interface
class GNNSceneEmbeddingNetwork_LearnedEdgeVector(GNNSceneEmbeddingNetwork_Inference):
    """
    Backward compatible version of the original model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Utility functions for easier model creation
def create_training_model(dataset, **model_kwargs):
    """
    Create a training model with appropriate parameters from dataset.
    
    Args:
        dataset: SceneGraphDataset instance
        **model_kwargs: Additional model parameters
        
    Returns:
        GNNSceneEmbeddingNetwork_Training: Training model
    """
    defaults = {
        'object_feature_dim': 518,
        'num_relations': dataset.get_num_relations(),
        'node_embedding_dim': 128,
        'edge_embedding_dim': 32,
        'clip_dim': 512
    }
    defaults.update(model_kwargs)
    
    return GNNSceneEmbeddingNetwork_Training(**defaults)


def create_inference_model(dataset, **model_kwargs):
    """
    Create an inference model with appropriate parameters from dataset.
    
    Args:
        dataset: SceneGraphDataset instance
        **model_kwargs: Additional model parameters
        
    Returns:
        GNNSceneEmbeddingNetwork_Inference: Inference model
    """
    defaults = {
        'object_feature_dim': 518,
        'num_relations': dataset.get_num_relations(),
        'node_embedding_dim': 128,
        'edge_embedding_dim': 32,
        'scene_embedding_dim': 32
    }
    defaults.update(model_kwargs)
    
    return GNNSceneEmbeddingNetwork_Inference(**defaults)
