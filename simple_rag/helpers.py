from pydantic import BaseModel
from typing import Union, List
from google.genai.types import EmbedContentResponse
import numpy as np
import os
from pathlib import Path
from typing import List, Optional
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel
from langchain_core.documents import Document
import lancedb
from google import genai
from dotenv import load_dotenv
from google.genai.types import EmbedContentConfig
from concurrent.futures import ThreadPoolExecutor

env_file = Path(__file__).parent.parent / ".env"

load_dotenv(env_file)

N_DIM = 768

class Embedding(BaseModel):
    text: str
    embedded_content: EmbedContentResponse
    
    @property
    def vector(self) -> np.ndarray:
        if isinstance(self.embedded_content, EmbedContentResponse):
            return np.array(self.embedded_content.embeddings[0].values)
        raise NotImplementedError

class FileContent(BaseModel):
    filepath: Path
    content: str


def markdown_splitter(
        data: List[FileContent],
        chunk_size: int, 
        overlap_size: int) -> List[Document]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=True,
    )

    md_splits = [md_splitter.split_text(fc.content) for fc in data]

    # Make sure we add the filename to the metadata
    for i, page in enumerate(md_splits):
        for split in page:
            split.metadata["filename"] = data[i].filepath.name

    # Flatten the list of lists
    md_splits = [split for sublist in md_splits for split in sublist]

    # Don't forget to contraint split size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(md_splits)



def read_files_as_object_array(directory_path: Path) -> List[FileContent]:
    """
    Reads all files in the specified directory and returns their contents as an array of objects.

    Each object contains the filename and the content of the file.

    Args:
        directory_path (str): Path to the directory containing the files.

    Returns:
        list: A list of dictionaries, where each dictionary has 'filename' and 'content' keys.
    """
    object_array: List[str] = []

    # Iterate through all files in the directory
    for filepath in directory_path.iterdir():
        # Skip directories, process only files
        if filepath.is_file():
            try:
                content = filepath.read_text(encoding="utf-8")
                filecontent = FileContent(filepath=filepath, content=content)
                object_array.append(filecontent)
            except Exception as e:
                print(f"Error reading file {filepath.name}: {e}")

    return object_array

class EmbeddingGenerator:
    client = genai.Client(vertexai=True)

    def generate(self, doc: str, title: Optional[str] = None) -> Embedding:
        response =  self.client.models.embed_content(
            model="text-embedding-005",
            contents=[doc,], # this is bcz it accepts only list
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",  # Optional
                output_dimensionality=N_DIM,  # Optional
                title=title,  # Optional
            ), 
        )
        return Embedding(text=doc, embedded_content=response)
    

# lancedb_dir = data_dir / "lance_db_data"
DB_URI = os.getenv("LANCE_DB_URI", ".data/lance_db_data")