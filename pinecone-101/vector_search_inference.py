import configparser
from pinecone import Pinecone as PineconeClient


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
        PineconeClient: Pinecone client.
    """
    return PineconeClient(api_key=api_key, source_tag='pinecone-notebooks:pinecone-101')


def get_embedding(pc_client, input_question):
    """
    Get the embedding of an input question using an external model, and then process it with the Pinecone client.

    Args:
        pc_client (PineconeClient): The Pinecone client used for embedding operations.
        input_question (str): The input question or text for which the embedding will be generated.

    Returns:
        list: The embedding vector for the input question.
    """
    res = pc_client.inference.embed(
        model="multilingual-e5-large",
        inputs=input_question,
        parameters={
            "input_type": "query",  # or "passage"
            "truncate": "END"
        }
    )
    embedding = res.data[0]['values']
    return embedding


def search_index(pc, index_name, search_vector):
    """
    Search the Pinecone index using the provided search vector.

    Args:
        pc (PineconeClient): Pinecone client.
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
    index_name = "pinecone101-inference"

    # Initialize standard Pinecone client
    pc_client = create_pinecone_client(api_key)

    search_text = "what if my patient has knee pain"
    search_vector = get_embedding(pc_client, search_text)
    search_index(pc_client, index_name, search_vector)
