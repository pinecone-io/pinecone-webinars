import configparser
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch


def load_configuration(config_file='config.ini'):
    """
    Load configuration from a specified file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        configparser.ConfigParser: Loaded configuration.
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def create_pinecone_client(api_key):
    """
    Create a Pinecone client with the provided API key.

    Args:
        api_key (str): Pinecone API key.

    Returns:
        Pinecone: Pinecone client.
    """
    return Pinecone(api_key=api_key, source_tag='pinecone-notebooks:pinecone-101')


def get_embedding(text):
    """
    Get embeddings for the given text using Hugging Face transformer model.

    Args:
        text (str): The input text.

    Returns:
        list: The embeddings for the text.
    """
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**tokenized_input)

    embedding = model_output.last_hidden_state[0].mean(dim=0)
    return embedding.tolist()


def search_index(pc, index_name, search_vector):
    """
    Search the Pinecone index using the provided search vector.

    Args:
        pc (Pinecone): Pinecone client.
        index_name (str): Name of the index.
        search_vector (list): The search vector.

    Returns:
        None
    """
    index = pc.Index(index_name)
    result = index.query(
        vector=search_vector,
        top_k=3,
        include_metadata=True
    )

    num_matches = len(result.get('matches', []))
    print(f"Number of matches: {num_matches}")
    print(f"Search vector (first 5 values): {search_vector[:5]}")
    print(f"Full search results: {result}")


if __name__ == "__main__":
    config = load_configuration()
    api_key = config['DEFAULT']['PINECONE_API_KEY']
    index_name = "pinecone101"

    pc = create_pinecone_client(api_key)

    search_text = "what if my patient has stomach pain"
    search_vector = get_embedding(search_text)

    search_index(pc, index_name, search_vector)
