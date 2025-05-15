import os
from pathlib import Path
from typing import List
import logging
import pandas as pd
import pyarrow
import numpy as np


import lancedb
from dotenv import load_dotenv
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_fixed


from helpers import Embedding, markdown_splitter, read_files_as_object_array, FileContent, EmbeddingGenerator, DB_URI
from helpers import N_DIM
# Import and initialize Vertex AI
import vertexai
vertexai.init()


table_name = "ai_library_documentation"

# Configure logging
current_file = Path(__file__)
main_dir = current_file.parent.parent
log_file = main_dir / ".logs" / f"{current_file.stem}.log"

data_dir = main_dir / ".data"
documentation_dir = data_dir / "documentations"

# Ensure the file handler captures DEBUG logs
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add the file handler to the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(logging.StreamHandler())

# Retry logic for embedding generation
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_embedding_with_retry(text: str, embed: EmbeddingGenerator) -> Embedding:
    return embed.generate(text)

def generate_embeddings_parallel(texts: List[str]) -> List[Embedding]:
    """Generate embeddings for a list of texts in parallel with retry logic."""
    embed = EmbeddingGenerator()

    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(lambda text: generate_embedding_with_retry(text, embed), texts))

    return embeddings

# Add logger statements to track progress and errors
logging.info("Reading files from the documentation directory...")
filecontents: List[FileContent] = read_files_as_object_array(documentation_dir)
logging.info(f"Read {len(filecontents)} files from the documentation directory.")

logging.info("Splitting files into smaller chunks...")
splits = markdown_splitter(filecontents, 3000, 100)
logging.info(f"Split files into {len(splits)} chunks.")

# Prepare an array of string from the documents
splits_as_string = []
for doc in splits:
    source_file = f"Source File: {doc.metadata.get('filename')}\n" if doc.metadata.get('filename') else ""
    headers = [f"{v} ({k})" for k, v in doc.metadata.items() if k not in ["filename"]]
    headers_str = "\n".join(headers) if headers else ""
    headers_str_full = f"\nExtract from the sections:\n{headers_str}" if headers_str else headers_str
    content_string_with_metadata_info = f"{source_file+headers_str_full}\n\n{doc.page_content}\n"
    splits_as_string.append(content_string_with_metadata_info)

logging.info("Generating embeddings for the document chunks...")
embeddings = generate_embeddings_parallel(splits_as_string)
logging.info(f"Generated embeddings for {len(embeddings)} chunks.")

logging.info("Connecting to the LanceDB database...")


db = lancedb.connect(uri=DB_URI)
logging.info("Connected to the database.")



# Prepare data with unique hashes
def generate_content_hash(content: str) -> str:
    """Generate a unique hash for the content."""
    return sha256(content.encode('utf-8')).hexdigest()

# Remove duplicate data initialization
# Define data once and reuse it
logging.info("Preparing data for upsert operations...")
data = [
    {
        "vector": embedding.vector,
        "content": embedding.text,
        "hash": generate_content_hash(embedding.text),
    }
    for embedding in embeddings
]
logging.info(f"Prepared {len(data)} records for upsert operations.")

# Correctly handle upsert logic without recreating the table
# Check if the table already exists
try:
    tbl = db.open_table(table_name)
    logging.info(f"Table '{table_name}' already exists. Proceeding with upsert operations.")
except ValueError:
    logging.info(f"Table '{table_name}' does not exist. Creating a new table...")
    # schema = {
    #     "vector": "vector",
    #     "content": "string",
    #     "hash": "string",
    # }
    schema = pyarrow.schema(
        [
            pyarrow.field("vector", pyarrow.list_(pyarrow.float32(), N_DIM)),
            pyarrow.field("content", pyarrow.string()),                # String for content
            pyarrow.field("hash", pyarrow.string()), 
        ]
    )
    tbl = db.create_table(table_name, schema=schema)



# Retrieve existing data from the table
existing_data = tbl.to_pandas()

# Prepare a DataFrame for new data
data_df = pd.DataFrame(data)

# Merge existing and new data, prioritizing new data for duplicate hashes
merged_data = pd.concat([existing_data, data_df]).drop_duplicates(subset="hash", keep="last")

# Debug log to verify duplicate handling
logging.debug("Checking for duplicate hashes...")
logging.debug(f"Existing data hashes: {existing_data['hash'].tolist()}")
logging.debug(f"New data hashes: {data_df['hash'].tolist()}")
logging.debug(f"Merged data hashes: {merged_data['hash'].tolist()}")

# Update the table with the merged data
if not merged_data.equals(existing_data):
    logging.info("Data has changed. Updating the table...")
    tbl.add(merged_data.to_dict(orient="records"))
    logging.info("Table updated successfully.")
else:
    logging.info("No changes detected. Table remains unchanged.")

