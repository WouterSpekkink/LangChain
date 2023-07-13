from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
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
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Please try to give detailed answers. Give appropriate citations of literature and include a bibliography for those citations.

{context}

Question: {question}"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
  
# Choose prompt
prompt = CONDENSE_QUESTION_PROMPT
prompt = PROMPT

# Set up conversational chain
chain = ConversationalRetrievalChain.from_llm(
  llm = ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=db.as_retriever(),
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
    source = source.replace('/home/wouter/Documents/LangChain/data/', '')
    source = source.replace('/home/wouter/Documents/LangChain/data/new/','')
    source = source.replace('/home/wouter/Documents/LangChain/data/old/','')
    print(source)

  print("\n")
  chat_history.append((query, result['answer']))
  query = None
