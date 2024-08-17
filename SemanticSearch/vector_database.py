from typing import List, Dict
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma as ChromaVecStore
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient
from directory_parser import parse_folder_to_doc


def index_directory(doc_path: str,
                    chunk_size: int = 1000,
                    chunk_overlap: int = 500,
                    chroma_path: str = "chroma",
                    collection_name: str = "my_collection") -> None:
    """
    Parses all files of a directory (currenltly only pdf supported). Indexes and adds them to a chroma database.

    :param doc_path: path to folder to be indexed
    :param chunk_size: size of produced text chunks
    :param chunk_overlap: overlap between produced text chunks
    :param chroma_path: Path to persistent chroma-db
    :param collection_name: Name of collection to used. Created in case it does not exist

    :return: None
    """

    chunks = parse_folder_to_doc(doc_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    save_to_chroma(chunks, chroma_path=chroma_path, collection_name=collection_name)


def save_to_chroma(chunks: List[Document], chroma_path: str = "chroma", collection_name: str = "my_collection") -> None:
    """
    Saves list of documents in chunks to chromadb saved at path chroma_path.

    :param chunks: List of documents to be added
    :param chroma_path: Path to persistent chroma-db
    :param collection_name: Name of collection to used. Created in case it does not exist

    :return: None
    """

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

    # Add data
    ids = [chunk.id for chunk in chunks]

    num_elems_pre = client.get_collection(name=collection_name).count()
    print(f"#Entries in {collection_name}: {num_elems_pre} before insert")
    vector_store.add_documents(chunks, ids=ids)
    num_elems_post = client.get_collection(name=collection_name).count()
    print(f"#Entries in {collection_name}: {client.get_collection(name=collection_name).count()} after insert")
    print(f"Added {num_elems_post - num_elems_pre} elements")
