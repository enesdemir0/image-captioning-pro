import pytest
import os
from src.data.dataset_loader import DataLoader

def test_data_loader_initialization(config):
    """Checks if the DataLoader can be initialized with the config."""
    loader = DataLoader(config)
    assert loader.image_dir == config['dataset']['image_dir']
    assert loader.caption_file == config['dataset']['caption_file']

# This test only runs if you have the JSON file locally
@pytest.mark.skipif(not os.path.exists("data/annotations/captions_train2014.json"), 
                    reason="MS-COCO annotations not found locally")
def test_load_annotations_logic(config):
    """Verifies that the parser actually finds captions in the JSON."""
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    
    assert len(captions) > 0
    assert "<start>" in captions[0]
    assert "<end>" in captions[0]