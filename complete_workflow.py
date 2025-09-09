#!/usr/bin/env python3
"""
Complete workflow demonstration for Scene Graph Masked Pre-training.

This script shows the entire pipeline:
1. Data loading and validation
2. Multi-task masked pre-training 
3. Transfer to inference model
4. Generation of final [1, 32] scene embeddings
"""

import torch
import numpy as np
import os
from pathlib import Path

# Import our modules
from dataset import SceneGraphDataset
from models import create_training_model, create_inference_model
from multitask_training import MultiTaskMaskedTrainer
from data_utils import create_sample_data, validate_directory


def step1_setup_data(data_path: str = "./data", num_samples: int = 50):
    """
    Step 1: Setup and validate data
    """
    print("🔧 STEP 1: Setting up data")
    print("-" * 40)
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Check if we have data, create samples if needed
    pkl_files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    
    if len(pkl_files) == 0:
        print(f"No .pkl files found in {data_path}")
        print(f"Creating {num_samples} sample files for demonstration...")
        create_sample_data(data_path, num_samples=num_samples)
        print(f"✓ Created {num_samples} sample .pkl files")
    else:
        print(f"Found {len(pkl_files)} .pkl files in {data_path}")
    
    # Validate data
    print("\nValidating data format...")
    validate_directory(data_path)
    
    # Load dataset
    dataset = SceneGraphDataset(data_path, relations_txt="/home/rizo/mipt_ccm/scene_encoder/data/vlsat_relations.txt")
    print(f"\n✓ Dataset loaded successfully:")
    print(f"  - {len(dataset)} scenes")
    print(f"  - {dataset.get_num_relations()} unique relations")
    print(f"  - Relations: {list(dataset.get_relation_mapping().keys())}")
    
    return dataset


def step2_pretrain_encoder(dataset: SceneGraphDataset, 
                          epochs: int = 50,
                          save_path: str = "checkpoints/pretrained_encoder.pth"):
    """
    Step 2: Multi-task masked pre-training
    """
    print(f"\n🎓 STEP 2: Multi-task masked pre-training")
    print("-" * 40)
    
    # Create training model
    model = create_training_model(
        dataset,
        node_embedding_dim=128,
        edge_embedding_dim=32,
        clip_dim=512
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Training model: {total_params:,} parameters")
    
    # Create trainer with the exact specifications
    trainer = MultiTaskMaskedTrainer(
        model=model,
        dataset=dataset,
        node_mask_ratio=0.15,        # 15% node masking
        edge_mask_ratio=0.15,        # 15% edge masking
        task_switch_probability=0.5   # 50-50 task switching
    )
    
    print(f"Starting masked pre-training...")
    print(f"  - Task A: Masked Node Prediction (CLIP reconstruction)")
    print(f"  - Task B: Masked Edge Prediction (Relation prediction)")
    print(f"  - {epochs} epochs with task switching")
    
    # Train
    trainer.train(
        num_epochs=epochs,
        batch_size=4,
        save_path=save_path,
        log_every=10
    )
    
    print(f"✓ Pre-training completed, model saved to {save_path}")
    return model, save_path


def step3_create_inference_model(dataset: SceneGraphDataset,
                                trained_model,
                                scene_embedding_dim: int = 32):
    """
    Step 3: Create inference model and transfer weights
    """
    print(f"\nSTEP 3: Creating inference model")
    print("-" * 40)
    
    # Create inference model for downstream tasks
    inference_model = create_inference_model(
        dataset,
        scene_embedding_dim=scene_embedding_dim
    )
    
    # Transfer pre-trained encoder weights
    inference_model.load_encoder_weights(trained_model)
    
    print(f"✓ Inference model created:")
    print(f"  - Scene embedding dimension: {scene_embedding_dim}")
    print(f"  - Pre-trained encoder weights transferred")
    
    return inference_model


def step4_generate_scene_embeddings(inference_model, dataset: SceneGraphDataset):
    """
    Step 4: Generate final scene embeddings for downstream tasks
    """
    print(f"\nSTEP 4: Generating scene embeddings")
    print("-" * 40)
    
    inference_model.eval()
    
    # Generate embeddings for all scenes
    scene_embeddings = []
    
    print("Generating scene embeddings...")
    with torch.no_grad():
        for i in range(min(10, len(dataset))):  # First 10 scenes for demo
            data = dataset[i]
            scene_embedding = inference_model(data)
            scene_embeddings.append(scene_embedding.squeeze().numpy())
            
            if i == 0:
                print(f"  Sample embedding shape: {scene_embedding.shape}")
    
    scene_embeddings = np.array(scene_embeddings)
    print(f"✓ Generated embeddings for {len(scene_embeddings)} scenes")
    print(f"  Final shape: {scene_embeddings.shape}")  # Should be [num_scenes, 32]
    
    # Show some statistics
    print(f"\nEmbedding Statistics:")
    print(f"  Mean: {scene_embeddings.mean():.4f}")
    print(f"  Std: {scene_embeddings.std():.4f}")
    print(f"  Min: {scene_embeddings.min():.4f}")
    print(f"  Max: {scene_embeddings.max():.4f}")
    
    return scene_embeddings


def step5_downstream_task_demo(scene_embeddings: np.ndarray):
    """
    Step 5: Demonstrate how to use scene embeddings for downstream tasks
    """
    print(f"\nSTEP 5: Downstream task demonstration")
    print("-" * 40)
    
    print("Your [1, 32] scene embeddings are now ready for:")
    print("  1. Scene classification")
    print("  2. Scene retrieval")
    print("  3. Scene similarity computation")
    print("  4. Clustering and analysis")
    
    # Example: Compute pairwise similarities
    if len(scene_embeddings) > 1:
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(scene_embeddings)
        print(f"\nExample - Pairwise cosine similarities:")
        print(f"  Shape: {similarities.shape}")
        print(f"  Average similarity: {similarities.mean():.4f}")
        
        # Find most similar scenes
        np.fill_diagonal(similarities, -1)  # Remove self-similarity
        most_similar_idx = np.unravel_index(np.argmax(similarities), similarities.shape)
        print(f"  Most similar scenes: {most_similar_idx[0]} & {most_similar_idx[1]}")
        print(f"  Similarity score: {similarities[most_similar_idx]:.4f}")
    
    print(f"\n✓ Scene embeddings ready for your specific downstream task!")


def main():
    """
    Complete workflow demonstration
    """
    print("=" * 60)
    print("SCENE GRAPH MASKED PRE-TRAINING WORKFLOW")
    print("=" * 60)
    print()
    print("This script demonstrates the complete pipeline:")
    print("1. Data setup and validation")
    print("2. Multi-task masked pre-training") 
    print("3. Inference model creation")
    print("4. Scene embedding generation")
    print("5. Downstream task preparation")
    print()
    
    # Configuration
    DATA_PATH = "./data"
    EPOCHS = 30  # Reduced for demo, use 100+ for real training
    SCENE_EMBEDDING_DIM = 32
    
    try:
        # Step 1: Setup data
        dataset = step1_setup_data(DATA_PATH, num_samples=40)
        
        # Step 2: Pre-train encoder
        trained_model, model_path = step2_pretrain_encoder(dataset, epochs=EPOCHS)
        
        # Step 3: Create inference model
        inference_model = step3_create_inference_model(
            dataset, trained_model, SCENE_EMBEDDING_DIM
        )
        
        # Step 4: Generate scene embeddings
        scene_embeddings = step4_generate_scene_embeddings(inference_model, dataset)
        
        # Step 5: Downstream task demo
        step5_downstream_task_demo(scene_embeddings)
        
        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Key Outputs:")
        print(f"Pre-trained model: {model_path}")
        print(f"Scene embeddings: {scene_embeddings.shape}")
        print(f"Ready for downstream tasks!")
        print()
        print("Next Steps:")
        print("  1. Use scene_embeddings for your specific task")
        print("  2. Fine-tune inference_model end-to-end if needed")
        print("  3. Scale up training with more data and epochs")
        
    except Exception as e:
        print(f"\n❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
