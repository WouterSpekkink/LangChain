from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.vectorstores import Chroma

import os
import sys
from dotenv import load_dotenv
import openai
import constants

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Get query as argument
query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

folder_path = "/home/wouter/Tools/Zotero/storage/"

PERSIST = True

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = DirectoryLoader(folder_path)
    if PERSIST:
        index = VectorstoreIndexCreator(
            embedding=OpenAIEmbeddings(),
            text_splitter=CharacterTextSplitter(chunk_size=5000, chunk_overlap=0),
            vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator(
            embedding=OpenAIEmbeddings(),
            text_splitter=CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)).from_loaders(loader)

# Setup chain to get responses from
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(),
  max_tokens_limit=4000)

# Set up conversation
chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None

