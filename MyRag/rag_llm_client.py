from typing import Dict
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def create_prompt(query: str, db_query_result: Dict) -> str:
    """
    Creates prompt from query and vector db search results.
    Taken mostly from from https://github.com/pixegami/rag-tutorial-v2/blob/main/query_data.py

    :param query: The query to be answered using information from db
    :param db_query_result: The information from db

    :return: The prompt
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


def prompt_openai(prompt: str, generator_llm: str):
    """
    Generate text using prompt and openAI model given by generator llm

    :param prompt: The prompt for generation
    :param generator_llm: Name of the generator llm

    :return: None
    """

    print("************************************************************************")
    print(f"Model used: {generator_llm}")
    model = ChatOpenAI(model="gpt-4o-mini",
                       temperature=0,
                       max_tokens=None,
                       timeout=None,
                       max_retries=2,)
    print(f"Start generating for prompt: ")
    print(prompt)
    response_text = model.predict(prompt)
    print("**************************************")
    print("Generated answer")
    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(response_text)
    print("************************************************************************")

def prompt_llm(prompt: str, generator_llm: str) -> None:
    """
    Generate text using prompt and model given by generator llm

    :param prompt: The prompt for generation
    :param generator_llm: Name of the generator llm

    :return: None
    """

    if generator_llm == "GPT-4o mini":
        print("! Prompt OpenAI !")
        prompt_openai(prompt, generator_llm)
    else:
        generator = pipeline("text-generation", model=generator_llm)
        tokens = generator.tokenizer.encode(prompt, add_special_tokens=False)

        print("************************************************************************")
        print(f"Model used: {generator_llm}")
        print(f"#Tokens in prompt: {len(tokens)}")
        print(f"Model max length: {generator.tokenizer.model_max_length}")
        print(f"Start generating")
        response_text = generator(prompt, max_length=generator.tokenizer.model_max_length, truncation=True,
                                  num_return_sequences=1)
        print("**************************************")
        print("Generated answer")
        print(response_text[0]["generated_text"])
        print("************************************************************************")


def query_llm_rag(query: str, db_query_result: Dict, generator_llm: str = "distilgpt2") -> None:
    """
    Query llm using data retrieved from vector database

    :param query: The query to be answered using information from db
    :param db_query_result: The information from db
    :param generator_llm: Name of generator llm. Must be in transformers lib

    :return: None
    """

    prompt = create_prompt(query, db_query_result)
    prompt_llm(prompt, generator_llm)
