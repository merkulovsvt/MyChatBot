from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from datasets import load_dataset

model_name = 'hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q8_0'
# dataset = load_dataset("wikimedia/wikipedia", "20231101.ru")

model = Ollama(model=model_name, temperature=0.7)
embedding_model = OllamaEmbeddings(model=model_name)

retriever = Chroma(embedding_function=embedding_model,
                   persist_directory="./chroma_db").as_retriever(search_type="similarity", k=2)
rag_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True
)


def get_rag_response(messages):
    return rag_chain.invoke(messages[-1]['content'])["result"]
