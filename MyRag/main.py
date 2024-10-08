from yaml import safe_load
from typing import Dict
from vector_database import index_directory, query_database
from rag_llm_client import query_llm_rag

def load_config(config_file: str = "config.yaml") -> Dict:
    """
    Loads yaml config and returns a dict

    :param config_file: Path to config file


    :return: The config object
    """

    with open(config_file, "r") as config_file:
        config = safe_load(config_file)

    return config

def query_llm_rag_main():
    config = load_config()
    chroma_path = config["default_chroma_path"]
    collection_name = config["default_collection_name"]
    num_result_entries = config["num_result_entries_from_db"]
    generator_llm = config["generator_llm"]
    query = "How does Alice meet the Mad Hatter?"

    db_query_result = query_database(query,
                                     num_result_entries=num_result_entries,
                                     chroma_path=chroma_path,
                                     collection_name=collection_name)

    query_llm_rag(query, db_query_result, generator_llm=generator_llm)

def query_db_main():
    config = load_config()
    chroma_path = config["default_chroma_path"]
    collection_name = config["default_collection_name"]
    num_result_entries = config["num_result_entries_from_db"]
    query = "How does Alice meet the Mad Hatter?"

    db_query_result = query_database(query,
                                     num_result_entries=num_result_entries,
                                     chroma_path=chroma_path,
                                     collection_name=collection_name)

    return db_query_result




def index_dir_main():
    config = load_config()
    doc_path = "../DataSets"
    chunk_size = config["default_chunk_size"]
    chunk_overlap = config["default_chunk_overlap"]
    chroma_path = config["default_chroma_path"]
    collection_name = config["default_collection_name"]
    embedding_type = config["database_embedding"]

    index_directory(doc_path,
                    embedding_type=embedding_type,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chroma_path=chroma_path,
                    collection_name=collection_name)


if __name__ == '__main__':
    #index_dir_main()
    #query_db_main()
    query_llm_rag_main()