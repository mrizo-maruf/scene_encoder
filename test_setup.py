#!/usr/bin/env python3
"""
Test script to verify the scene graph dataset implementation.
"""

import os
import sys
import traceback

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Geometric import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    print("All imports successful!\n")
    return True


def test_dataset_creation():
    """Test creating sample data and loading with dataset."""
    print("Testing dataset creation...")
    
    try:
        from data_utils import create_sample_data, validate_directory
        from dataset import SceneGraphDataset
        
        # Create test data
        test_data_dir = "./data"
        create_sample_data(test_data_dir, num_samples=3)
        
        # Validate the created data
        print("\nValidating created data:")
        validate_directory(test_data_dir)
        
        # Test dataset loading
        print("\nTesting dataset loading:")
        dataset = SceneGraphDataset(test_data_dir, relations_txt="/home/rizo/mipt_ccm/scene_encoder/data/vlsat_relations.txt")
        print(f"✓ Dataset created with {len(dataset)} samples")
        print(f"✓ Found {dataset.get_num_relations()} unique relations")
        print(f"✓ Relations: {list(dataset.get_relation_mapping().keys())}")
        
        # Test getting a sample
        if len(dataset) > 0:
            data = dataset[0]
            print(f"✓ Sample 0 - Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
            print(f"✓ Node features shape: {data.x.shape}")
            print(f"✓ Edge index shape: {data.edge_index.shape}")
            print(f"✓ Edge attributes shape: {data.edge_attr.shape}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_data_dir)
        print("✓ Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test creating and running both training and inference models."""
    print("\nTesting model creation...")
    
    try:
        from models import (
            create_training_model, 
            create_inference_model,
            GNNSceneEmbeddingNetwork_LearnedEdgeVector
        )
        from data_utils import create_sample_data
        from dataset import SceneGraphDataset
        import torch
        
        # Create sample dataset for testing
        test_data_dir = "./test_model_data"
        create_sample_data(test_data_dir, num_samples=3)
        dataset = SceneGraphDataset(test_data_dir, relations_txt="/home/rizo/mipt_ccm/scene_encoder/data/vlsat_relations.txt")

        # Test training model
        training_model = create_training_model(dataset, node_embedding_dim=64)
        print("✓ Training model created successfully")
        
        # Test inference model
        inference_model = create_inference_model(dataset, scene_embedding_dim=32)
        print("✓ Inference model created successfully")
        
        # Test backward compatibility
        legacy_model = GNNSceneEmbeddingNetwork_LearnedEdgeVector(
            object_feature_dim=518,
            num_relations=dataset.get_num_relations(),
            scene_embedding_dim=32
        )
        print("✓ Legacy model created successfully")
        
        # Test forward passes with dummy data
        sample_data = dataset[0]
        
        # Training model forward pass
        training_model.eval()
        with torch.no_grad():
            training_outputs = training_model(sample_data)
            print(f"✓ Training model forward pass successful")
            print(f"  Node embeddings: {training_outputs['node_embeddings'].shape}")
            if training_outputs['node_predictions'] is not None:
                print(f"  Node predictions: {training_outputs['node_predictions'].shape}")
        
        # Inference model forward pass
        inference_model.eval()
        with torch.no_grad():
            scene_embedding = inference_model(sample_data)
            print(f"✓ Inference model forward pass successful")
            print(f"  Scene embedding: {scene_embedding.shape}")
        
        # Legacy model forward pass
        legacy_model.eval()
        with torch.no_grad():
            legacy_output = legacy_model(sample_data)
            print(f"✓ Legacy model forward pass successful")
            print(f"  Legacy output: {legacy_output.shape}")
        
        # Test weight transfer
        inference_model.load_encoder_weights(training_model)
        print("✓ Weight transfer successful")
        
        # Clean up
        import shutil
        shutil.rmtree(test_data_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        traceback.print_exc()
        return False


def test_dataloader():
    """Test DataLoader functionality."""
    print("\nTesting DataLoader...")
    
    try:
        from data_utils import create_sample_data
        from dataset import SceneGraphDataset, create_dataloader
        
        # Create test data
        test_data_dir = "./test_data_loader"
        create_sample_data(test_data_dir, num_samples=5)
        
        # Create dataset and dataloader
        dataset = SceneGraphDataset(test_data_dir, relations_txt="/home/rizo/mipt_ccm/scene_encoder/data/vlsat_relations.txt")
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
        
        # Test batch
        batch = next(iter(dataloader))
        print(f"✓ DataLoader created successfully")
        print(f"✓ Batch nodes: {batch.x.shape[0]}")
        print(f"✓ Batch edges: {batch.edge_index.shape[1]}")
        print(f"✓ Batch size: {batch.batch.max().item() + 1}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_data_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        traceback.print_exc()
        return False


def test_masked_training():
    """Test the masked training functionality."""
    print("\nTesting masked training...")
    
    try:
        from masked_training import MaskedTrainer
        from models import create_training_model
        from data_utils import create_sample_data
        from dataset import SceneGraphDataset
        
        # Create test data
        test_data_dir = "./test_masked_data"
        create_sample_data(test_data_dir, num_samples=5)
        dataset = SceneGraphDataset(test_data_dir, relations_txt="/home/rizo/mipt_ccm/scene_encoder/data/vlsat_relations.txt")
        
        # Create training model
        model = create_training_model(dataset, node_embedding_dim=32)
        
        # Create trainer
        trainer = MaskedTrainer(
            model=model,
            dataset=dataset,
            device='cpu',  # Force CPU for testing
            node_mask_ratio=0.2,
            edge_mask_ratio=0.2
        )
        print("✓ Masked trainer created successfully")
        
        # Test single training step
        from torch_geometric.loader import DataLoader
        dataloader = DataLoader(dataset, batch_size=2)
        batch = next(iter(dataloader))
        
        losses = trainer.train_step(batch)
        print(f"✓ Training step successful")
        print(f"  Total loss: {losses['total_loss']:.4f}")
        print(f"  Node loss: {losses['node_loss']:.4f}")
        print(f"  Edge loss: {losses['edge_loss']:.4f}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_data_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Masked training test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Scene Graph Dataset Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Dataset Creation Test", test_dataset_creation),
        ("Model Creation Test", test_model_creation),
        ("DataLoader Test", test_dataloader),
        ("Masked Training Test", test_masked_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Your scene graph dataset is ready to use.")
        print("\nNext steps:")
        print("1. Add your .pkl files to the ./data directory")
        print("2. Run: python data_utils.py --validate")
        print("3. Run: python example_usage.py")
        print("4. Run: python masked_training.py (for self-supervised pretraining)")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
