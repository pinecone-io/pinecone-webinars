import configparser
import openai
from pinecone import Pinecone


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


def create_pinecone_client(api_key: str) -> Pinecone:
    """
    Create a Pinecone client with the provided API key.

    Args:
        api_key (str): Pinecone API key.

    Returns:
        Pinecone: The Pinecone client.
    """
    return Pinecone(api_key=api_key, source_tag='pinecone-notebooks:pinecone-101')


def get_openai_embeddings(api_key: str, text: list, model: str = "text-embedding-ada-002") -> list:
    """
    Get embeddings for the given text using OpenAI.

    Args:
        api_key (str): OpenAI API key.
        text (list): List of texts to get embeddings for.
        model (str): OpenAI model to use for embeddings.

    Returns:
        list: List of embeddings.
    """
    openai.api_key = api_key
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    embeddings = [item.embedding for item in response.data]

    return embeddings


def search_index(pc: Pinecone, index_name: str, search_vector: list) -> None:
    """
    Search the Pinecone index using the provided search vector.

    Args:
        pc (Pinecone): Pinecone client.
        index_name (str): Name of the index.
        search_vector (list): The search vector.
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
    index_name = "pinecone101-openai"

    pc = create_pinecone_client(api_key)

    search = "what if my patient has stomach pain"
    openai_api_key = config['DEFAULT']['OPENAI_API_KEY']
    vector_search = get_openai_embeddings(openai_api_key, [search])

    search_index(pc, index_name, vector_search)
