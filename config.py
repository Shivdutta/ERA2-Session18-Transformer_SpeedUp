from pathlib import Path


def get_config() -> dict:
    """
    Function to return dictionary of model configuration
    """
    return {
        'batch_size': 2048,                       # Number of sentences
        'num_epochs': 10,
        'lr': 10**-4,
        'seq_len': 350,                           # Maximum number of words in data sample
        'd_model': 512,                           # Dimension of embeddings i.e. an array of 512 values in which some represent score for living, non-living, finance, nature, etc.
                                                  # Currently, this number is around 10_000 for GPT-4
        'lang_src': "en",                         # Source language: English
        'lang_tgt': "it",                         # Target language: Italian
        'model_folder': "weights",                # Folder in which weights will be stored
        'model_basename': 'tmodel_',              # Prefix
        'preload': False,                         # If True, it will start from existing saved model
        'tokenizer_file': "tokenizer_{0}.json",   # Name of a file created to save tokens
        'experiment_name': "runs/tmodel"          # Training logs directory
    }


def get_weights_file_path(config: dict, epoch: str) -> str:
    """
    Function to retrieve model from specific basename
    :param config: Configuration dictionary for the run
    :param epoch: Number of epoch whose model is to be retrieved
    :return: Path of the model file
    """
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
