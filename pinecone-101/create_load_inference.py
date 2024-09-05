import json
import configparser
import time
import pandas as pd
from pinecone import Pinecone as PineconeClient, ServerlessSpec


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
        PineconeClient: An instance of the Pinecone client.
    """
    return PineconeClient(api_key=api_key, source_tag='pinecone-notebooks:pinecone-101')


def create_index(pc, index_name):
    """
    Create a Pinecone index with the specified name if it doesn't already exist.

    Args:
        pc (PineconeClient): The Pinecone client.
        index_name (str): The name of the index to create.

    Returns:
        bool: True if the index already exists, False if a new index was created.
    """
    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exists.")
        return True
    else:
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Index '{index_name}' created.")
        return False


def get_sample_data(file_name):
    """
    Load and process sample data from a JSONL file.

    Args:
        file_name (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, each containing a record's data.
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


def get_data_pandas(pc_client, file_name):
    """
    Process sample data, retrieve embeddings for each record using the Pinecone client,
    and store the results in a pandas DataFrame.

    Args:
        pc_client (PineconeClient): The Pinecone client used to generate embeddings.
        file_name (str): The path to the JSONL file containing the sample data.

    Returns:
        pandas.DataFrame: A DataFrame containing the records with their embeddings and metadata.
    """
    sample_data = get_sample_data(file_name)
    vectors = []

    for record in sample_data:
        record_id = record["record_id"]
        text = record["note"]
        metadata = record["metadata"]

        embedding_values = get_embedding(pc_client, text)

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


def upsert_df(pc, index_name, records_df):
    """
    Upsert data from a pandas DataFrame into the specified Pinecone index.

    Args:
        pc (PineconeClient): The Pinecone client.
        index_name (str): The name of the index.
        records_df (pandas.DataFrame): The DataFrame containing the records to upsert.

    Returns:
        None
    """
    index = pc.Index(name=index_name)
    index.upsert_from_dataframe(records_df)

    print(f"Data successfully upserted into index '{index_name}'.")
    time.sleep(5)
    print(index.describe_index_stats())


if __name__ == "__main__":
    config = load_configuration()
    api_key = config['DEFAULT']['PINECONE_API_KEY']
    file_name = "data/demo_doctor_notes.jsonl"
    index_name = "pinecone101-inference"

    # Initialize standard Pinecone client
    pc_client = create_pinecone_client(api_key)

    if create_index(pc_client, index_name):
        exit()
    else:
        records_df = get_data_pandas(pc_client, file_name)  #XXXXXX
        upsert_df(pc_client, index_name, records_df)
        print("Data UPSERT completed.")
