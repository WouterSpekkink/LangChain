from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
import os
import sys
import constants
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Set HuggingFace key
os.environ['API_KEY'] = constants.HUGKEY

# Setup Falcon
model_id = 'tiiuae/falcon-40b-instruct'

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})

# Load FAISS database
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("./vectorstore/", embeddings)

# Get query as argument
query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

# Customize prompt
system_prompt_template = ("You are a knowledgeable professor working in academia.\n"
                          "Using the provided pieces of context, you answer the questions asked by the human.\n"
                          "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
                          "Please try to give detailed answers and write your answers as an academic text, unless explicitly told otherwise.\n"
                          "Use references to literature in your answer and include a bibliography for citations that you use.\n"
                          "Context: {context}")

system_prompt = PromptTemplate(template=system_prompt_template,
                               input_variables=["context"])

system_message_prompt = SystemMessagePromptTemplate(prompt = system_prompt)
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Set up conversational chain
chain = ConversationalRetrievalChain.from_llm(
  #llm = ChatOpenAI(model="gpt-3.5-turbo"),
  llm = falcon_llm,
  retriever=db.as_retriever(search_type="mmr", search_kwargs={"k" : 10}),
  chain_type="stuff",
  return_source_documents = True,
  combine_docs_chain_kwargs={'prompt': chat_prompt},
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
