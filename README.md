# Scene Graph Encoder

A PyTorch Geometric implementation for self-supervised scene graph representation learning using masked training. This project implements a modular GNN architecture that learns scene embeddings through multi-task masked pre-training on VL-SAT style scene graph data.

## 🏗️ Architecture

### Core Components

The architecture consists of three main components designed for flexible training and inference:

#### 1. **GNNSceneEncoder** (Shared Encoder)
- **Input**: Scene graph with 518D node features (bbox_center[3] + bbox_extent[3] + CLIP_descriptor[512])
- **Architecture**: 
  - Node encoder: Linear(518 → 256 → 128) with ReLU and Dropout
  - Edge embedding: Learnable embeddings for relation types
  - Graph layers: 2x GAT layers with edge features and multi-head attention
- **Output**: Node embeddings [num_nodes, 128]

#### 2. **Prediction Heads** (Training Only)
- **NodePredictionHead**: MLP(128 → 128 → 64 → 6) for bbox coordinate reconstruction
- **EdgePredictionHead**: MLP(256 → 128 → 64 → num_relations) for relation prediction

#### 3. **SceneEmbeddingHead** (Inference Only)  
- **Input**: Node embeddings from encoder
- **Architecture**: Global mean pooling + MLP(128 → 64 → 32)
- **Output**: Scene embedding [1, 32] for downstream tasks

### Model Configurations

```python
# Training Model (for masked pre-training)
GNNSceneEmbeddingNetwork_Training:
  ├── GNNSceneEncoder (shared)
  ├── NodePredictionHead (bbox reconstruction)
  └── EdgePredictionHead (relation prediction)

# Inference Model (for downstream tasks)
GNNSceneEmbeddingNetwork_Inference:
  ├── GNNSceneEncoder (shared, pre-trained)
  └── SceneEmbeddingHead (scene-level output)
```

## 🎓 Training: Multi-Task Masked Learning

### Training Strategy

The model uses a **multi-task masked training** approach that alternates between two self-supervised tasks:

#### Task A: Masked Node Prediction (50% probability)
1. **Masking**: Randomly select 15% of nodes and zero out their bbox coordinates (center + extent)
2. **Objective**: Reconstruct original bbox coordinates from corrupted node embeddings
3. **Loss**: MSELoss between predicted and original bbox coordinates

#### Task B: Masked Edge Prediction (50% probability)  
1. **Masking**: Randomly select 15% of edges, store their relation labels
2. **Objective**: Predict relation types from concatenated node embeddings
3. **Loss**: CrossEntropyLoss for relation classification

### Training Process

```python
# Multi-task training loop
for each batch:
    if random() < 0.5:  # Task A
        masked_data, mask, original_bbox = mask_nodes(data, ratio=0.15)
        node_embeddings = encoder(masked_data)
        predicted_bbox = node_head(node_embeddings[mask])
        loss = MSELoss(predicted_bbox, original_bbox)
    else:  # Task B
        masked_edges, relations = mask_edges(data, ratio=0.15)
        node_embeddings = encoder(data)
        edge_features = concat([h_i, h_j])  # source + target embeddings
        predicted_relations = edge_head(edge_features)
        loss = CrossEntropyLoss(predicted_relations, relations)
```

### Key Training Features

- **Task Switching**: 50-50 probability ensures balanced learning
- **Adaptive Masking**: At least 1 node/edge masked even in small graphs  
- **Gradient Clipping**: Prevents exploding gradients during training
- **Learning Rate Scheduling**: ReduceLROnPlateau for stable convergence
- **Model Checkpointing**: Saves best model based on combined loss

### Usage

```python
from multitask_training import MultiTaskMaskedTrainer
from models import create_training_model

# Create training model
model = create_training_model(dataset)

# Initialize trainer
trainer = MultiTaskMaskedTrainer(
    model=model,
    dataset=dataset,
    node_mask_ratio=0.15,
    edge_mask_ratio=0.15,
    task_switch_probability=0.5
)

# Train with multi-task objectives
trainer.train(num_epochs=100, batch_size=8)
```

## 🎯 Getting the Encoder (Without Decoder)

### Method 1: Extract from Training Model

After pre-training, you can extract just the encoder for feature extraction:

```python
from models import create_training_model

# Load pre-trained training model
model = create_training_model(dataset)
model.load_state_dict(torch.load('pretrained_model.pth')['model_state_dict'])

# Extract encoder only
encoder = model.encoder
encoder.eval()

# Use encoder for node embeddings
with torch.no_grad():
    node_embeddings = encoder(data)  # Shape: [num_nodes, 128]
```

### Method 2: Transfer to Inference Model

For downstream tasks requiring scene-level embeddings:

```python
from models import create_training_model, create_inference_model

# Load pre-trained training model
training_model = create_training_model(dataset)
training_model.load_state_dict(torch.load('pretrained_model.pth')['model_state_dict'])

# Create inference model and transfer encoder weights
inference_model = create_inference_model(dataset, scene_embedding_dim=32)
inference_model.load_encoder_weights(training_model)

# Generate scene embeddings
with torch.no_grad():
    scene_embedding = inference_model(data)  # Shape: [1, 32]
```

### Method 3: Save Encoder Separately

```python
# During or after training
torch.save({
    'encoder_state_dict': model.encoder.state_dict(),
    'model_config': {
        'object_feature_dim': 518,
        'num_relations': dataset.get_num_relations(),
        'node_embedding_dim': 128,
        'edge_embedding_dim': 32
    }
}, 'encoder_only.pth')

# Load encoder later
from models import GNNSceneEncoder

config = torch.load('encoder_only.pth')['model_config']
encoder = GNNSceneEncoder(**config)
encoder.load_state_dict(torch.load('encoder_only.pth')['encoder_state_dict'])
```

## 🚀 Quick Start

### 1. Data Preparation
```python
from dataset import SceneGraphDataset

# Your .pkl files should contain:
# {
#   'objects': [
#     {
#       'node_id': 'unique_id',
#       'bbox_center': [x, y, z],
#       'bbox_extent': [w, h, d], 
#       'clip_descriptor': [512D vector],
#       'edges_vl_sat': [list of edges]
#     }
#   ]
# }

dataset = SceneGraphDataset('./data', relations_txt='./vlsat_relations.txt')
```

### 2. Training
```python
# Complete workflow
python complete_workflow.py

# Or step by step
python multitask_training.py  # Multi-task pre-training
python example_usage.py       # Model testing and validation
```

### 3. Inference
```python
# Generate scene embeddings for downstream tasks
inference_model = create_inference_model(dataset)
inference_model.load_encoder_weights(trained_model)

scene_embedding = inference_model(data)  # [1, 32] ready for classification/retrieval
```

## 📊 Model Performance

The trained encoder learns to:
- ✅ Reconstruct bbox coordinates from corrupted inputs (spatial understanding)
- ✅ Predict spatial relations from node pairs (edge-level reasoning)  
- ✅ Generate informative scene embeddings (scene-level representation)

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📁 File Structure

```
scene_encoder/
├── dataset.py              # VL-SAT data loading and preprocessing
├── models.py               # Modular GNN architecture (encoder + heads)
├── multitask_training.py   # Multi-task masked training implementation
├── masked_training.py      # Alternative single-task training
├── complete_workflow.py    # End-to-end pipeline demonstration
├── example_usage.py        # Model testing and validation
├── data_utils.py          # Data utilities and validation
└── requirements.txt       # Dependencies
```

## 🎯 Use Cases

1. **Scene Classification**: Use [1, 32] embeddings for scene categorization
2. **Scene Retrieval**: Compute similarities between scene embeddings  
3. **Few-shot Learning**: Fine-tune inference model on downstream tasks
4. **Feature Extraction**: Use node embeddings [num_nodes, 128] for object-level analysis

The modular design allows flexible deployment: use the encoder for node-level features or the full inference model for scene-level representations, depending on your downstream task requirements.