from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import langchain
import os
import glob
from dotenv import load_dotenv
import openai
import constants
import time

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Load documents
folder_path = '/home/wouter/Documents/LangChain/data/new'
print("===Loading documents===")
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(folder_path,
                         show_progress=True,
                         use_multithreading=True,
                         loader_cls=TextLoader,
                         loader_kwargs=text_loader_kwargs)
documents = loader.load()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    add_start_index = True,
)

split_documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    show_progress_bar=True,
    request_timeout=60,
)

print("===Embedding text and creating database===")
new_db = FAISS.from_documents(split_documents, embeddings)
print("===Merging new and old database===")
old_db = FAISS.load_local("./vectorstore/", embeddings)
merged_db = old_db.merge_from(new_db)
merged_db.save_local("./vectorstore/", "index")

