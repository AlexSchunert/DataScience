from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from typing import List, Dict
from hashlib import sha256

def add_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Adds sha256 to input chunks

    :param param1: Input chunks

    :return: Chunks with ids
    """

    for chunk in chunks:
        #unique_id = chunk.metadata["source"] + str(chunk.metadata["page"]) + str(chunk.metadata["start_index"])
        hash_obj = sha256()
        hash_obj.update(chunk.page_content.encode())
        #unique_id = hash_obj.hexdigest()
        chunk.id = hash_obj.hexdigest()


    return chunks


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
    chunks = add_chunk_ids(chunks)

    print(f"Split {len(docs)} docs into {len(chunks)} chunks")
    return chunks
