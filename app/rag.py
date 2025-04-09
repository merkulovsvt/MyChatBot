import os
import json
from usearch.index import Index

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.usearch import USearch
from langchain_community.docstore.in_memory import InMemoryDocstore

llm = OllamaLLM(model="hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q8_0", temperature=0.3)
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(f"{project_root}/rag/usearch/docstore.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

docstore_data = {k: Document(**v) for k, v in raw_data.items()}
docstore = InMemoryDocstore(docstore_data)

with open(f"{project_root}/rag/usearch/ids.json") as f:
    ids = json.load(f)

index = Index()
index.load(f"{project_root}/rag/usearch/index.usearch")

usearch = USearch(embedding=embedding_model, index=index, docstore=docstore, ids=ids)

retriever = usearch.as_retriever(search_type="similarity", k=10)

rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)


def get_rag_response(messages):
    result = rag_chain.invoke(messages[-1]['content'])
    return result["result"]
