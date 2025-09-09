import os
import pickle
import torch
from torch_geometric.data import Dataset, Data
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict


class SceneGraphDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset for scene graph data from .pkl files.
    
    Each .pkl file should contain:
    - detections: list of objects with bbox_center, bbox_extent, clip_descriptor, id
    - edges_vl_sat: list of edges with relation information
    """
    
    def __init__(self, root: str, relations_txt:str, transform=None, pre_transform=None, pre_filter=None):
        """
        Initialize the SceneGraphDataset.
        
        Args:
            root (str): Root directory containing the .pkl files
            transform: Optional transform to be applied on data objects
            pre_transform: Optional pre-transform to be applied on data objects
            pre_filter: Optional pre-filter to filter out data objects
        """
        self.root = root
        self.pkl_files = []
        self.relation_to_idx = {}
        # self._build_relation_mapping_vlsat()
        # Load relationship names and check the file start
        
        with open(relations_txt, "r") as f:
            self.relationships_list = [line.strip() for line in f.readlines() if line.strip()!='']

        self.vlsat_rel_id_to_rel_name = {i: name for i, name in enumerate(self.relationships_list)}
        # print(f"DEBUG: Loaded {len(self.relationships_list)} relationships from {relations_txt}, mapping: {self.vlsat_rel_id_to_rel_name}")
        # print(f"DEBUG: Loaded {len(self.relationships_list)} relationships from {relations_txt}")
        super().__init__(root, transform, pre_transform, pre_filter)
        
    def _build_relation_mapping_vlsat(self):
        """
        Build a global mapping from relation strings to integers.
        This scans all .pkl files to create a comprehensive mapping.
        """
        print("Building relation mapping...")
        all_relations = set()
        
        # Scan all .pkl files to collect unique relations
        pkl_files = [f for f in os.listdir(self.root) if f.endswith('.pkl')]
        
        # print(f"DEBUG: Found {len(pkl_files)} .pkl files for relation mapping.")
        for pkl_file in pkl_files:
            file_path = os.path.join(self.root, pkl_file)
            # print(f"DEBUG: Processing {file_path}")
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    data = data['objects']
                    # print(f'DEBUG: Loaded .pkl file with keys: {data[0].keys()}')

                for object in data:
                    for edge in object.get('edges_vl_sat', []):
                        print(f"DEBUG: Edge keys: {edge.keys()}")
                        if isinstance(edge, dict) and 'relation' in edge:
                            all_relations.add(edge['relation'])
                        elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
                            # Assuming format: [source_id, target_id, relation]
                            all_relations.add(edge[2])
                            
            except Exception as e:
                print(f"Warning: Could not process {pkl_file}: {e}")
                continue
        
        # Create mapping
        self.relation_to_idx = {relation: idx for idx, relation in enumerate(sorted(all_relations))}
        print(f"Found {len(self.relation_to_idx)} unique relations: {list(self.relation_to_idx.keys())}")
        
        # Add unknown relation for safety
        if 'unknown' not in self.relation_to_idx:
            self.relation_to_idx['unknown'] = len(self.relation_to_idx)
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        return [f for f in os.listdir(self.root) if f.endswith('.pkl')]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        # We don't pre-process files, so return empty list
        return []
    
    def download(self):
        """Download raw data. Not needed for local .pkl files."""
        pass
    
    def process(self):
        """Process raw data. Not needed as we process on-the-fly."""
        pass
    
    def len(self) -> int:
        """Return the number of data samples."""
        return len(self.raw_file_names)
    
    def get(self, idx: int) -> Data:
        """
        Get a single data sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        pkl_file = self.raw_file_names[idx]
        file_path = os.path.join(self.root, pkl_file)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"DEBUG: Loaded {len(data['objects'])} objects from {pkl_file}, type: {type(data)}, keys: {data.keys()}")
            data = data['objects']

        return self._convert_to_torch_geometric(data, pkl_file)
    
    def _convert_to_torch_geometric(self, data: Dict[str, Any], filename: str) -> Data:
        """
        Convert raw data to PyTorch Geometric Data object.
        
        Args:
            data (dict): Raw data from .pkl file
            filename (str): Name of the source file
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        # Extract detections
        edges_vl_sat = []
        
        # Build node features and ID mapping
        node_features = []
        
        for idx, node in enumerate(data):
            # Extract features
            bbox_center = self._safe_extract_feature(node, 'bbox_center', 3)
            bbox_extent = self._safe_extract_feature(node, 'bbox_extent', 3)
            clip_descriptor = self._safe_extract_feature(node, 'clip_descriptor', 512)

            # Concatenate features [3 + 3 + 512 = 518]
            features = np.concatenate([bbox_center, bbox_extent, clip_descriptor])
            node_features.append(features)
            
            # Build ID mapping
            obj_id = node.get('node_id', f'obj_{idx}')
        
        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Process edges
        edge_index, edge_attr = self._process_edges_vlsat(edges_vl_sat)

        # Create Data object
        torch_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            filename=filename,
            num_nodes=len(data)
        )
        
        return torch_data
    
    def _safe_extract_feature(self, detection: Dict[str, Any], key: str, expected_size: int) -> np.ndarray:
        """
        Safely extract and validate feature from detection.
        
        Args:
            detection (dict): Detection data
            key (str): Feature key
            expected_size (int): Expected feature size
            
        Returns:
            np.ndarray: Feature array of correct size
        """
        feature = detection.get(key, None)
        
        if feature is None:
            # Return zeros if feature is missing
            return np.zeros(expected_size)
        
        # Convert to numpy array
        feature = np.array(feature, dtype=np.float32)
        
        # Handle different shapes
        if feature.ndim == 0:
            # Scalar
            feature = np.array([feature])
        elif feature.ndim > 1:
            # Flatten if multi-dimensional
            feature = feature.flatten()
        
        # Ensure correct size
        if len(feature) < expected_size:
            # Pad with zeros if too small
            feature = np.pad(feature, (0, expected_size - len(feature)))
        elif len(feature) > expected_size:
            # Truncate if too large
            feature = feature[:expected_size]
        
        return feature
    
    def _process_edges_vlsat(self, edges_vl_sat: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process edges into COO format.
        
        Args:
            edges_vl_sat (list): List of edges
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: edge_index and edge_attr tensors
        """
        edge_sources = []
        edge_targets = []
        edge_relations = []
        
        for edge in edges_vl_sat:
            try:
                # Dictionary format: {'source': id, 'target': id, 'relation': str}
                source_id = edge['id_1']
                target_id = edge['id_2']
                relation = edge['rel_name']
                relation_id = edge['rel_id']
            
                
                edge_sources.append(source_id)
                edge_targets.append(target_id)
                edge_relations.append(relation_id)
                
            except Exception as e:
                print(f"Warning: Error processing edge {edge}: {e}")
                continue
        
        # Create tensors
        if edge_sources:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
            edge_attr = torch.tensor(edge_relations, dtype=torch.long)
        else:
            # Empty graph
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.long)
        
        return edge_index, edge_attr
    
    def get_relation_mapping(self) -> Dict[str, int]:
        """Return the relation to index mapping."""
        return self.vlsat_rel_id_to_rel_name
    
    def get_num_relations(self) -> int:
        """Return the number of unique relations."""
        return len(self.vlsat_rel_id_to_rel_name.values())


def create_dataloader(dataset: SceneGraphDataset, batch_size: int = 32, shuffle: bool = True, **kwargs):
    """
    Create a DataLoader for the SceneGraphDataset.
    
    Args:
        dataset (SceneGraphDataset): The dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader: PyTorch Geometric DataLoader
    """
    from torch_geometric.loader import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


# Example usage and testing functions
def test_dataset(data_path: str):
    """
    Test the dataset implementation.
    
    Args:
        data_path (str): Path to the directory containing .pkl files
    """
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist")
        return
    
    print(f"Testing dataset with data from: {data_path}")
    
    # Create dataset
    dataset = SceneGraphDataset(data_path)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of unique relations: {dataset.get_num_relations()}")
    print(f"Relation mapping: {dataset.get_relation_mapping()}")
    
    if len(dataset) > 0:
        # Test first sample
        data = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Node features shape: {data.x.shape}")
        print(f"  Edge index shape: {data.edge_index.shape}")
        print(f"  Edge attributes shape: {data.edge_attr.shape}")
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Number of edges: {data.edge_index.shape[1]}")
        print(f"  Filename: {data.filename}")
        
        # Test dataloader
        dataloader = create_dataloader(dataset, batch_size=2)
        batch = next(iter(dataloader))
        print(f"\nBatch test:")
        print(f"  Batch size: {batch.batch.max() + 1}")
        print(f"  Total nodes in batch: {batch.x.shape[0]}")
        print(f"  Total edges in batch: {batch.edge_index.shape[1]}")


if __name__ == "__main__":
    # Example usage
    test_dataset("./data")
