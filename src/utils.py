# src/utils.py

import logging

def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        filename='../logs/pipeline.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_data_loading():
    """Unit test for data loading."""
    from preprocessing import load_data
    df = load_data('../data/hyperspectral_corn_data.csv')
    assert len(df) > 0, "Data loading failed"

def test_model_creation():
    """Unit test for model creation."""
    from train import create_nn
    model = create_nn(100)
    assert model is not None, "Model creation failed"

if __name__ == '__main__':
    setup_logging()
    test_data_loading()
    test_model_creation()