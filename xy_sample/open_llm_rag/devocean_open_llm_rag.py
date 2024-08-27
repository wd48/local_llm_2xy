# https://devocean.sk.com/community/detail.do?ID=166016&boardType=DEVOCEAN_STUDY

import os
from langchain.document_loaders import TextLoader

import warnings
warnings.filterwarnings("ignore")

################################
## TextLoader
################################

from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("2024.kbo.rules.pdf")
pages = loader.load()

text = pages[0].page_content

import tiktoken

# create sentence embedding, vector DB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

################################
## TextSplitter
################################

# separate document to sentence
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

docs = text_splitter.split_documents(pages)

################################
## Text Vector Embedding
################################

# convert sentence to embedding, save to vector database
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'mps'},
    encode_kwargs={'normalize_embeddings':True},
)

################################
## VectorStore(Chroma)
################################

vectorstore = Chroma.from_documents(docs, embeddings)

# vector store path
vectorstore_path = 'vectorstore'

# create & save vector store
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
vectorstore.persist()
print("Vectorstore created and persisted")

################################
## Retriever
################################