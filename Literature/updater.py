from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import langchain
import os
import shutil
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
source_path = './data/new'
destination_path = './data/old'
store_path = './vectorstore/'
print("===Loading documents===")
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(source_path,
                         show_progress=True,
                         use_multithreading=True,
                         loader_cls=TextLoader,
                         loader_kwargs=text_loader_kwargs)
documents = loader.load()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 150,
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
old_db = FAISS.load_local(store_path, embeddings)
old_db.merge_from(new_db)
old_db.save_local(store_path, "index")

# Move files
for filename in os.listdir(source_path):
    # construct full file path
    source = os.path.join(source_path, filename)
    destination = os.path.join(destination_path, filename)
    
    # move the file
    shutil.move(source, destination)
