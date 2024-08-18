from yaml import safe_load
from typing import Dict
from vector_database import index_directory, query_database


def load_config(config_file: str = "config.yaml") -> Dict:
    """
    Loads yaml config and returns a dict

    :param config_file: Path to config file


    :return: The config object
    """

    with open(config_file, "r") as config_file:
        config = safe_load(config_file)

    return config


def query_db_main():
    config = load_config()
    chroma_path = config["default_chroma_path"]
    collection_name = config["default_collection_name"]
    query = "What is science"

    query_database(chroma_path,
                   collection_name,
                   query)


def index_dir_main():
    config = load_config()
    doc_path = "../DataSets"
    chunk_size = config["default_chunk_size"]
    chunk_overlap = config["default_chunk_overlap"]
    chroma_path =  config["default_chroma_path"]
    collection_name = config["default_collection_name"]

    index_directory(doc_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chroma_path=chroma_path,
                    collection_name=collection_name)


if __name__ == '__main__':
    #index_dir_main()
    query_db_main()
