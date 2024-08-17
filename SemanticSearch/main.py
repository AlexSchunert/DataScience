import fitz  # PyMuPDF
from vector_database import index_directory





def main():
    doc_path = "../DataSets"
    chunk_size = 100
    chunk_overlap = 50
    chroma_path = "./chroma"
    collection_name = "my_collection"

    index_directory(doc_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chroma_path=chroma_path,
                    collection_name=collection_name)



if __name__ == '__main__':
    main()
