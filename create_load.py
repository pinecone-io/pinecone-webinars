import json
import configparser
import time

from pinecone import Pinecone, ServerlessSpec

from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd


def load_configuration(config_file='config.ini'):
    """
    Load the configuration from a specified file.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        configparser.ConfigParser: The loaded configuration.
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def create_pinecone_client(api_key):
    """
    Create a Pinecone client with the provided API key.

    Args:
        api_key (str): The Pinecone API key.

    Returns:
        Pinecone: The Pinecone client.
    """
    return Pinecone(api_key=api_key, source_tag='pinecone-notebooks:pinecone-101')


def get_sample_data(file_name):
    """
    Get sample data from a JSONL file.

    Args:
        file_name (str): The path to the JSONL file.

    Returns:
        list: A list of records.
    """
    records = []
    with open(file_name, 'r') as file:
        for record_number, line in enumerate(file, start=1):
            entry = json.loads(line)
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


def get_embedding(text):
    """
    Get embeddings for the given text using Hugging Face transformer model.

    Args:
        text (str): The input text.

    Returns:
        torch.Tensor: The embeddings for the text.
    """
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    embedding = model_output.last_hidden_state[0].mean(dim=0)
    return embedding


def get_data(file_name):
    """
    Get data from the specified file.

    Args:
        file_name (str): The path to the JSONL file.

    Returns:
        list: A list of vectors containing embeddings and metadata.
    """
    sample_data = get_sample_data(file_name)
    vectors = []

    for record in sample_data:
        record_id = record["record_id"]
        text = record["note"]
        metadata = record["metadata"]

        embedding_values = get_embedding(text)

        vector = {
            "id": record_id,
            "values": embedding_values,
            "metadata": metadata
        }

        vectors.append(vector)

    return vectors


def get_data_pandas(file_name):
    """
    Process data from the specified file and return a pandas DataFrame.

    Args:
        file_name (str): The path to the JSONL file.

    Returns:
        pandas.DataFrame: The processed data in a DataFrame.
    """
    sample_data = get_sample_data(file_name)
    vectors = []

    for record in sample_data:
        record_id = record["record_id"]
        text = record["note"]
        metadata = record["metadata"]

        embedding_values = get_embedding(text).tolist()

        vector = {
            "id": record_id,
            "values": embedding_values,
            "metadata": metadata
        }

        vectors.append(vector)

    df = pd.DataFrame(vectors)

    print(df)
    df.to_json('data/sample_notes_data.jsonl', orient='records', lines=True)

    return df


def upsert_data(pc, index_name, vectors):
    """
    Upsert data into the specified Pinecone index.

    Args:
        pc (Pinecone): The Pinecone client.
        index_name (str): The name of the index.
        vectors (list): The vectors to upsert.
    """
    if index_name in pc.list_indexes().names:
        index = pc.Index(name=index_name)
        index.upsert(vectors=vectors)
        print(f"Data successfully upserted into index '{index_name}'.")
        print(index.describe_index_stats())


def upsert_df(pc, index_name, records_df):
    """
    Upsert data from a pandas DataFrame into the specified Pinecone index.

    Args:
        pc (Pinecone): The Pinecone client.
        index_name (str): The name of the index.
        records_df (pandas.DataFrame): The DataFrame containing the records to upsert.
    """
    index = pc.Index(name=index_name)
    index.upsert_from_dataframe(records_df)

    print(f"Data successfully upserted into index '{index_name}'.")
    time.sleep(5)
    print(index.describe_index_stats())


def create_index(pc, index_name):
    """
    Create a Pinecone index if it does not already exist.

    Args:
        pc (Pinecone): The Pinecone client.
        index_name (str): The name of the index.

    Returns:
        bool: True if the index already exists, False otherwise.
    """
    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exists.")
        return True
    else:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Index '{index_name}' created.")
        return False


if __name__ == "__main__":
    config = load_configuration()
    api_key = config['DEFAULT']['PINECONE_API_KEY']
    file_name = "data/demo_doctor_notes.jsonl"

    index_name = "pinecone101-openai"
    pc = create_pinecone_client(api_key)

    if create_index(pc, index_name):
        exit()
    else:
        records_df = get_data_pandas(file_name)
        upsert_df(pc, index_name, records_df)
        print("Data UPSERT completed.")
