import fitz  # PyMuPDF
from vector_database import index_directory, query_database


def query_db_main():
    chroma_path = "./chroma"
    collection_name = "my_collection"
    query = "What is science"
    query_database(chroma_path,
                   collection_name,
                   query)


def index_dir_main():
    doc_path = "../DataSets"
    chunk_size = 600
    chunk_overlap = 300
    chroma_path = "./chroma"
    collection_name = "my_collection"

    index_directory(doc_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chroma_path=chroma_path,
                    collection_name=collection_name)



if __name__ == '__main__':
    #index_dir_main()
    query_db_main()
