from typing import List
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from os.path import exists
from shutil import rmtree
from yaml import safe_load as safe_load_yaml

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
    print(f"Split {len(docs)} docs into {len(chunks)} chunks")
    return chunks


def save_to_chroma(chunks: list[Document], chroma_path: str = "chroma") -> None:



    with open('config.yaml', 'r') as file:
        config = safe_load_yaml(file)
    api_key = config['openai_api_key']
    #embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Clear out the database first.
    if exists(chroma_path):
        rmtree(chroma_path)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=chroma_path, collection_name="example"
    )
    #db.persist()
    print(f"Saved {len(chunks)} chunks to {chroma_path}.")
