
# Pinecone Starter Project

## Overview
This project demonstrates how to use Pinecone along with embedding models from Hugging Face and OpenAI to process, embed, and search textual data. The project includes scripts for data processing, embedding generation, and upserting data into a Pinecone index.

## Main Files
- `config.ini`: Configuration file containing the Pinecone API key and OpenAI API key.
- `procedural_create_load.py`: Script for loading data, creating embeddings using the Hugging Face `sentence-transformers/all-MiniLM-L6-v2` model, and upserting data to Pinecone.
- `procedural_vector_search.py`: Script for executing a vector search query based on a user inputted query/question using the Hugging Face embedding model.
- `pinecone101.ipynb`: Notebook to demonstrate the end-to-end process of loading data, creating embeddings, and executing a vector search query.
- `demo_doctor_notes.jsonl`: Sample data file containing doctor notes for demonstration purposes.

## Supplemental Files

- `procedural_create_load_openai.py`: Script for loading data, creating embeddings using the OpenAI `text-embedding-ada-002`, and upserting data to Pinecone.
- `procedural_vector_openai.py`: Script for executing a vector search query based on a user inputted query/question using the OpenAI embedding model.
- `sample_data_generator.py`: Script to generate sample data for demonstration purposes.


demo_doctor_notes.jsonl
## Requirements
- Python 3.8 or higher
- `pinecone-client`
- `transformers`
- `torch`
- `pandas`

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/brickj/pinecone_starter_v1.git
    ```
2. Install the required packages:
    
    Setup virtual environment and install the required python packages listed under requirements. 


## Configuration
1. Set your Pinecone API key in `config.ini`:
    ```ini
    PINECONE_API_KEY = your_pinecone_api_key
    ```

## Scripts Usage
### Step 1 
Run the main ```procedural_create_load.py``` script to get and embedd the demo_doctor_notes.jsonl and UPSERT data:
```bash
python create_load.py 
```

### Step 2
Run the main ```procedural_vector_search.py``` script to execute a vector search query:
```bash
python vector_search.py
```

## Notebooks Usage
### Step 1
Run the main ```pinecone101.ipynb``` notebook to get and embedd the demo_doctor_notes.jsonl, UPSERT data and execute a vector search query:

## Contact
For questions or support, please contact Rick Jacobs at rick.jacobs@example.com.
