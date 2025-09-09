import pickle
import numpy as np
from typing import Dict, List, Any
import os


def create_sample_data(output_path: str, num_samples: int = 5):
    """
    Create sample .pkl files for testing the dataset.
    
    Args:
        output_path (str): Directory to save sample files
        num_samples (int): Number of sample files to create
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Sample relations
    relations = ['on', 'left of', 'right of', 'behind', 'in front of', 'above', 'below', 'near']
    
    for i in range(num_samples):
        # Create random detections
        num_objects = np.random.randint(3, 8)  # 3-7 objects per scene
        detections = []
        
        for j in range(num_objects):
            detection = {
                'id': f'obj_{i}_{j}',
                'bbox_center': np.random.randn(3).tolist(),  # 3D center
                'bbox_extent': np.random.rand(3).tolist(),   # 3D extent
                'clip_descriptor': np.random.randn(512).tolist()  # CLIP features
            }
            detections.append(detection)
        
        # Create random edges
        edges_vl_sat = []
        num_edges = np.random.randint(2, min(10, num_objects * 2))
        
        for _ in range(num_edges):
            source_idx = np.random.randint(0, num_objects)
            target_idx = np.random.randint(0, num_objects)
            
            if source_idx != target_idx:  # Avoid self-loops
                edge = {
                    'source': f'obj_{i}_{source_idx}',
                    'target': f'obj_{i}_{target_idx}',
                    'relation': np.random.choice(relations)
                }
                edges_vl_sat.append(edge)
        
        # Create data dictionary
        data = {
            'detections': detections,
            'edges_vl_sat': edges_vl_sat,
            'metadata': {
                'scene_id': f'scene_{i}',
                'num_objects': num_objects,
                'num_edges': len(edges_vl_sat)
            }
        }
        
        # Save to file
        filename = f'sample_scene_{i:03d}.pkl'
        filepath = os.path.join(output_path, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Created {filename} with {num_objects} objects and {len(edges_vl_sat)} edges")
    
    print(f"Created {num_samples} sample files in {output_path}")


def validate_pkl_file(filepath: str) -> Dict[str, Any]:
    """
    Validate a .pkl file to ensure it has the correct format.
    
    Args:
        filepath (str): Path to the .pkl file
        
    Returns:
        Dict[str, Any]: Validation results
    """
    results = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            results['errors'].append("Data is not a dictionary")
            return results
        
        # Check for required keys
        if 'detections' not in data:
            results['errors'].append("Missing 'detections' key")
        
        if 'edges_vl_sat' not in data:
            results['errors'].append("Missing 'edges_vl_sat' key")
        
        # Validate detections
        detections = data.get('detections', [])
        if not isinstance(detections, list):
            results['errors'].append("'detections' is not a list")
        else:
            results['stats']['num_detections'] = len(detections)
            
            for i, detection in enumerate(detections):
                if not isinstance(detection, dict):
                    results['errors'].append(f"Detection {i} is not a dictionary")
                    continue
                
                # Check required fields
                required_fields = ['bbox_center', 'bbox_extent', 'clip_descriptor']
                for field in required_fields:
                    if field not in detection:
                        results['warnings'].append(f"Detection {i} missing '{field}'")
                    else:
                        # Check dimensions
                        value = detection[field]
                        if field in ['bbox_center', 'bbox_extent']:
                            if not (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3):
                                results['warnings'].append(f"Detection {i} '{field}' should have 3 dimensions")
                        elif field == 'clip_descriptor':
                            if not (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 512):
                                results['warnings'].append(f"Detection {i} '{field}' should have 512 dimensions")
                
                # Check for ID
                if 'id' not in detection:
                    results['warnings'].append(f"Detection {i} missing 'id' field")
        
        # Validate edges
        edges = data.get('edges_vl_sat', [])
        if not isinstance(edges, list):
            results['errors'].append("'edges_vl_sat' is not a list")
        else:
            results['stats']['num_edges'] = len(edges)
            
            # Collect object IDs for validation
            object_ids = set()
            for detection in detections:
                if isinstance(detection, dict) and 'id' in detection:
                    object_ids.add(detection['id'])
            
            for i, edge in enumerate(edges):
                if isinstance(edge, dict):
                    # Dictionary format
                    source = edge.get('source', edge.get('src', edge.get('from')))
                    target = edge.get('target', edge.get('dst', edge.get('to')))
                    relation = edge.get('relation', edge.get('rel'))
                    
                    if source is None:
                        results['warnings'].append(f"Edge {i} missing source ID")
                    elif source not in object_ids:
                        results['warnings'].append(f"Edge {i} references unknown source ID: {source}")
                    
                    if target is None:
                        results['warnings'].append(f"Edge {i} missing target ID")
                    elif target not in object_ids:
                        results['warnings'].append(f"Edge {i} references unknown target ID: {target}")
                    
                    if relation is None:
                        results['warnings'].append(f"Edge {i} missing relation")
                
                elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
                    # List/tuple format
                    source, target, relation = edge[0], edge[1], edge[2]
                    
                    if source not in object_ids:
                        results['warnings'].append(f"Edge {i} references unknown source ID: {source}")
                    if target not in object_ids:
                        results['warnings'].append(f"Edge {i} references unknown target ID: {target}")
                
                else:
                    results['warnings'].append(f"Edge {i} has unrecognized format")
        
        # If no errors, mark as valid
        if not results['errors']:
            results['valid'] = True
        
    except Exception as e:
        results['errors'].append(f"Failed to load file: {str(e)}")
    
    return results


def validate_directory(data_path: str):
    """
    Validate all .pkl files in a directory.
    
    Args:
        data_path (str): Path to directory containing .pkl files
    """
    if not os.path.exists(data_path):
        print(f"Directory {data_path} does not exist")
        return
    
    pkl_files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"No .pkl files found in {data_path}")
        return
    
    print(f"Validating {len(pkl_files)} .pkl files in {data_path}")
    print("=" * 50)
    
    total_valid = 0
    total_errors = 0
    total_warnings = 0
    
    for pkl_file in pkl_files:
        filepath = os.path.join(data_path, pkl_file)
        results = validate_pkl_file(filepath)
        
        print(f"\n{pkl_file}:")
        print(f"  Valid: {results['valid']}")
        
        if results['stats']:
            stats = results['stats']
            print(f"  Objects: {stats.get('num_detections', 0)}")
            print(f"  Edges: {stats.get('num_edges', 0)}")
        
        if results['errors']:
            print(f"  Errors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"    - {error}")
            total_errors += len(results['errors'])
        
        if results['warnings']:
            print(f"  Warnings ({len(results['warnings'])}):")
            for warning in results['warnings'][:5]:  # Show first 5 warnings
                print(f"    - {warning}")
            if len(results['warnings']) > 5:
                print(f"    ... and {len(results['warnings']) - 5} more")
            total_warnings += len(results['warnings'])
        
        if results['valid']:
            total_valid += 1
    
    print(f"\n" + "=" * 50)
    print(f"Summary:")
    print(f"  Total files: {len(pkl_files)}")
    print(f"  Valid files: {total_valid}")
    print(f"  Total errors: {total_errors}")
    print(f"  Total warnings: {total_warnings}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Utilities for scene graph data preparation")
    parser.add_argument('--create-samples', action='store_true', 
                       help='Create sample .pkl files for testing')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing .pkl files')
    parser.add_argument('--data-path', default='./data',
                       help='Path to data directory (default: ./data)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of sample files to create (default: 5)')
    
    args = parser.parse_args()
    
    if args.create_samples:
        print(f"Creating {args.num_samples} sample files in {args.data_path}")
        create_sample_data(args.data_path, args.num_samples)
    
    if args.validate:
        print(f"Validating files in {args.data_path}")
        validate_directory(args.data_path)
    
    if not args.create_samples and not args.validate:
        # Default behavior
        print("Scene Graph Data Utilities")
        print("Usage examples:")
        print("  python data_utils.py --create-samples")
        print("  python data_utils.py --validate")
        print("  python data_utils.py --create-samples --num-samples 10")
        print("  python data_utils.py --validate --data-path /path/to/data")
