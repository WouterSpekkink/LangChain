from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from datetime import datetime
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

# Load FAISS database
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("./vectorstore/", embeddings)

# Get query as argument
query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

# Set llm
llm = ChatOpenAI(model="gpt-3.5-turbo")
  
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
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

# Set retriever
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(base_compressor = pipeline_compressor, base_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k" : 10}))

# Set up conversational chain
chain = ConversationalRetrievalChain.from_llm(
  llm=llm,
  retriever=compression_retriever,
  chain_type="stuff",
  return_source_documents = True,
  combine_docs_chain_kwargs={'prompt': chat_prompt},
)

# Set up source file
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"sources/sources_{timestamp}.txt"
with open(filename, 'w') as file:
  file.write(f"Sources for search done on {timestamp}\n\n")

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
      with open(filename, 'a') as file:
        file.write("Query:\n")
        file.write(query)
        file.write("\n\n")
        file.write("Document: ")
        file.write(source.metadata['source'])
        file.write("\n\n")
        file.write("Content:\n")
        file.write(source.page_content)
        file.write("\n\n")
  for source in print_sources:
    source = os.path.basename(source)
    print(source)

  print("\n")
  chat_history.append((query, result['answer']))
  if (len(chat_history) > 3):
    chat_history.pop(0)
  query = None
