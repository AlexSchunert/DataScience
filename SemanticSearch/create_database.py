from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma as ChromaVecStore
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient
from dataclasses import dataclass
from os.path import exists
from shutil import rmtree
from yaml import safe_load as safe_load_yaml


def add_chunk_ids(chunks: List[Document]) -> List[Dict]:
    chunks_with_ids = []
    for chunk in chunks:
        unique_id = chunk.metadata["source"] + str(chunk.metadata["page"]) + str(chunk.metadata["start_index"])
        chunks_with_ids.append({"id": unique_id, "chunk": chunk})

    return chunks_with_ids


def load_docs(doc_path: str, chunk_size: int = 1000, chunk_overlap: int = 500) -> List[Document]:
    """
    Loads

    :param doc_path: path to folder with documents
    :param chunk_size: size of produced text chunks
    :param chunk_overlap: overlap between produced text chunks

    :return: list of the produced text chunks
    """
    loader = PyPDFDirectoryLoader(doc_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len,
                                                   add_start_index=True, )
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    # Add IDs
    chunks_with_ids = add_chunk_ids(chunks)

    print(f"Split {len(docs)} docs into {len(chunks)} chunks")
    return chunks


def save_to_chroma(chunks: list[Document], chroma_path: str = "chroma", collection_name: str = "my_collection") -> None:
    # Init embedding function
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Create or load chromaDB
    client = PersistentClient(path=chroma_path)
    # collection = client.get_or_create_collection(name=collection_name)

    vector_store = ChromaVecStore(client=client, embedding_function=embeddings, collection_name=collection_name)

    # Remove already

    # Add data
    vector_store.add_documents(chunks)

    print(f"Saved {len(chunks)} chunks to {chroma_path}.")
