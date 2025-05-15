# README

This repository contains code related to RAG (Retrieval-Augmented Generation). Currently, only a simple RAG implementation is available in the `simple_rag` directory.

## Instructions

1. **Run `download.py`**: This script downloads the necessary data for the RAG pipeline.
2. **Run `ingest.py`**: This script processes the downloaded data and ingests it into the database.
3. **Run `retrieve.py`**: This script retrieves data from the database and provides answers to user queries. Note that `retrieve.py` must be run in interactive mode because it contains asynchronous code that is not encapsulated within an `async def` function. Instead, `await` is called directly in the script.

### Example
To run `retrieve.py` in interactive mode, use the following command:

```bash
python -i simple_rag/retrieve.py
```

## Additional Information

The documentation files are downloaded from their respective documentation locations (e.g., Pydantic AI) using the `download.py` script. After downloading, the `ingest.py` script performs vector embedding updates to prepare the data for retrieval operations.