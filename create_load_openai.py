import json
import configparser
import openai
from pinecone import Pinecone, ServerlessSpec
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


# Create Pinecone client
def create_pinecone_client(api_key):
    """
    Create a Pinecone client with the provided API key.

    Args:
        api_key (str): Pinecone API key.

    Returns:
        Pinecone: The Pinecone client.
    """
    return Pinecone(api_key=api_key)


# Define the function to get sample data
def get_sample_data(file_name):
    """
    Get sample data from the specified file.

    Args:
        file_name (str): The name of the file containing sample data.

    Returns:
        list: A list of records with processed metadata.
    """
    record_number = 0
    records = []

    with open(file_name, 'r') as file:
        for line in file:
            record_number += 1
            entry = json.loads(line)

            # Handle null values in metadata and ensure it's a dictionary
            metadata = entry.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}

            for key, value in metadata.items():
                if value is None:
                    metadata[key] = "not available"

            record = {
                "record_number": record_number,
                "record_id": entry['patient_id'],
                "note": entry['note'],
                "metadata": metadata
            }
            records.append(record)

    return records


# Define the function to get OpenAI embeddings
def get_openai_embeddings(api_key, text, model="text-embedding-ada-002"):
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
    # Properly extract the embeddings from the response
    embeddings = [item.embedding for item in response.data]

    return embeddings


def get_embedding(text):
    """
    Get embeddings for the given text using a Hugging Face model.

    Args:
        text (str): The text to get embeddings for.

    Returns:
        torch.Tensor: The embedding tensor.
    """
    # Load the tokenizer and model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input question
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Generate the embedding
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Get the embedding from the model output
    embedding = model_output.last_hidden_state[0].mean(dim=0)
    return embedding


# Define the function to process and print data
def get_data(file_name):
    """
    Process data from the specified file and get embeddings.

    Args:
        file_name (str): The name of the file containing sample data.

    Returns:
        list: A list of vectors with embeddings and metadata.
    """
    config = load_configuration()
    openai_api_key = config['DEFAULT']['OPENAI_API_KEY']
    sample_data = get_sample_data(file_name)

    vectors = []

    for record in sample_data:
        record_id = record["record_id"]
        text = record["note"]
        metadata = record["metadata"]

        # Using the OpenAI API to get embeddings
        embeddings = get_openai_embeddings(openai_api_key, [text])
        embedding_values = embeddings[0]

        # # Using the Hugging Face model to get embeddings
        # embedding_values = get_embedding(text)

        vector = {
            "id": record_id,
            "values": embedding_values,
            "metadata": metadata
        }

        vectors.append(vector)

    return vectors


def upsert_data(pc, index_name, vectors):
    """
    Upsert data into the specified index.

    Args:
        pc (Pinecone): Pinecone client.
        index_name (str): Name of the index.
        vectors (list): List of vectors to upsert.
    """
    if index_name in pc.list_indexes().names():
        index = pc.Index(name=index_name)
        index.upsert(
            vectors=vectors
            #namespace="ns1"
        )
        print(f"Data successfully upserted into index '{index_name}'.")
        print(index.describe_index_stats())


def create_index(index_name):
    """
    Create a new Pinecone index if it does not already exist.

    Args:
        index_name (str): Name of the index to create.

    Returns:
        bool: True if the index already exists, False otherwise.
    """
    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exists.")
        return True
    else:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )


if __name__ == "__main__":
    config = load_configuration()
    api_key = config['DEFAULT']['PINECONE_API_KEY']
    openai_api_key = config['DEFAULT']['OPENAI_API_KEY']
    file_name = "data/demo_doctor_notes.jsonl"
    vectors = get_data(file_name)

    index_name = "pinecone101-openai"

    pc = create_pinecone_client(api_key)

    if create_index(index_name):
        exit()

    vectors = get_data(file_name)
    upsert_data(pc, index_name, vectors)
