from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
import sys
import constants
import openai

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Load FAISS database
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("./vectorstore/", embeddings)

# Get query as argument
query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

# Customize prompt
prompt_template = """Use the provided pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Please try to give detailed answers and write your answers as an academic text, unless explicitly told otherwise.

Always try to include appropriate citations of literature in your answer and always tell me if you are not able to do so. Always include a bibliography for citations that you use. 

Context: {context}

Question: {question}"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
prompt = PROMPT

# Set up conversational chain
chain = ConversationalRetrievalChain.from_llm(
  llm = ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=db.as_retriever(search_type="mmr", search_kwargs={"k" : 10}),
  chain_type="stuff",
  return_source_documents = True,
  combine_docs_chain_kwargs={'prompt': prompt},
)

# Set up conversation
chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print("\nAnswer:\n")
  print(result['answer'])
  print("\nSources:\n")
  sources = result['source_documents']
  print_sources = []
  for source in sources:
    if source.metadata['source'] not in print_sources:
      print_sources.append(source.metadata['source'])
  for source in print_sources:
    source = os.path.basename(source)
    print(source)

  print("\n")
  chat_history.append((query, result['answer']))
  if (len(chat_history) > 3):
    chat_history.pop(0)
  query = None
