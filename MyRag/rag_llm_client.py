from typing import Dict
from transformers import pipeline
from langchain.prompts import PromptTemplate


def create_prompt(query: str, db_query_result: Dict) -> str:
    """
    TBD
    Taken from https://github.com/pixegami/rag-tutorial-v2/blob/main/query_data.py

    :param query: TBD
    :param db_query_result: TBD

    :return: TBD
    """

    prompt_template_raw = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    prompt_template = PromptTemplate.from_template(prompt_template_raw)
    context = "\n\n---\n\n".join(db_query_result["documents"][0])
    prompt = prompt_template.format(context=context, question=query)

    return prompt

def query_llm_rag(query: str, db_query_result: Dict) -> None:
    """
    TBD

    :param query: TBD
    :param db_query_result: TBD

    :return: TBD
    """

    prompt = create_prompt(query, db_query_result)
    generator = pipeline('text-generation', model='distilgpt2')
    response_text = generator(prompt, max_length=10000, num_return_sequences=1)
    print(response_text)
    pass
