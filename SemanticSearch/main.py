import fitz  # PyMuPDF
from vector_database import save_to_chroma
from directory_parser import load_docs




def main():
    doc_path = "../DataSets"
    chunk_size = 100
    chunk_overlap = 50

    documents = load_docs(doc_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    save_to_chroma(documents, chroma_path="./chroma")
    pass
    # print(read_pfd("../DataSets/LectureNotesPhilScience.pdf"))
    # load_docs(doc_path)


if __name__ == '__main__':

    main()
