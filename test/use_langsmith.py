import openai
from token import OP
from openai import OpenAI, api_key, base_url

openai_client = OpenAI()

# This is the retriever we will use in RAG
# This is mocked out, but it could be anything we want
def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

# This is the end-to-end RAG chain.


# model = openai_client.models.retrieve(model="Qwen/Qwen3-32B",
#                                       api_key="sk-ypjnechwutigvbyglbkguukzsmzkkxfibauydwkbjrypwojd",
#                                       base_url="https://api.siliconflow.cn/v1")
        # model="Qwen/Qwen3-32B",
        # api_key="sk-ypjnechwutigvbyglbkguukzsmzkkxfibauydwkbjrypwojd",
        # base_url="https://api.siliconflow.cn/v1",
# It does a retrieval step then calls OpenAI
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
        {docs}""".format(docs="\n".join(docs))

    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model = "Qwen/Qwen3-32B",
    )
