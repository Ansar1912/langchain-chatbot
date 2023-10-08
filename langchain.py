# -*- coding: utf-8 -*-
# ! pip install langchain
# ! pip install openai
# ! pip install pypdf

import os
import openai
import langchain

openai.api_key  = "Enter Your OPEN AI KEY"

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("Agile-PM-101-Beginners-Guide-Non-PM-Ebook-download-open.pdf")
pages = loader.load()

len(pages)

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("Agile-PM-101-Beginners-Guide-Non-PM-Ebook-download-open.pdf"),
    PyPDFLoader("Agile-PM-101-Beginners-Guide-Non-PM-Ebook-download-open.pdf"),
    PyPDFLoader("2020-Scrum-Guide-US.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

len(splits)

#! pip install chromadb
#!pip install tiktoken

from langchain.vectorstores import Chroma

import os
import openai
import langchain

openai.api_key  = "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"

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)

persist_directory = 'my/chroma/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

docs = vectordb.similarity_search("What is agile development",k=3)

print(docs[0])

vectordb.persist()

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'my/chroma/'
embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

print(vectordb._collection.count())

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0,openai_api_key=openai.api_key)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

result = qa_chain({"query": question})

result["result"]

from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)

question="What is Kanban?"

result = qa_chain_mr({"query": question})

result["result"]

