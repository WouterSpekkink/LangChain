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
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import OpenAICallbackHandler
from datetime import datetime
import chainlit as cl
import textwrap
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

# Cleanup function for source strings
def string_cleanup(string):
  """A function to clean up strings in the sources from unwanted symbols"""
  return string.replace("{","").replace("}","").replace("\\","").replace("/","")

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Load FAISS database
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("./vectorstore/", embeddings)

# Set up callback handler
handler = OpenAICallbackHandler()

# Set up source file
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"answers/answers_{timestamp}.txt"
with open(filename, 'w') as file:
  file.write(f"Answers and sources for session started on {timestamp}\n\n")

@cl.on_chat_start
def main():
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
  
  # Set memory
  memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True, k = 3)
 
  # Set up conversational chain
  chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_type="mmr", search_kwargs={"k" : 10}),
    chain_type="stuff",
    return_source_documents = True,
    return_generated_question = True,
    combine_docs_chain_kwargs={'prompt': chat_prompt},
    memory=memory,
  )
  cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: str):
  chain = cl.user_session.get("chain")
  cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
  cb.answer_reached = True
  res = await chain.acall(message, callbacks=[cb])
  question = res["question"]
  answer = res["answer"]
  answer += "\n\n Sources:\n\n"
  sources = res["source_documents"]
  print_sources = []
  for source in sources:
    if source.metadata['source'] not in print_sources:
      print_sources.append(source.metadata['source'])
      with open(filename, 'a') as file:
        reference = "INVALID REF"
        if source.metadata.get('ENTRYTYPE') == 'article':
          reference = (
            string_cleanup(source.metadata.get('author', "")) + " (" +
            string_cleanup(source.metadata.get('year', "")) + "). " +
            string_cleanup(source.metadata.get('title', "")) + ". " +
            string_cleanup(source.metadata.get('journal', "")) + ", " +
            string_cleanup(source.metadata.get('volume', "")) + " (" +
            string_cleanup(source.metadata.get('number', "")) + "): " + 
            string_cleanup(source.metadata.get('pages', "")) + ".")
        elif source.metadata.get('ENTRYTYPE') == 'book':
          author = ""
          if 'author' in source.metadata:
            author = string_cleanup(source.metadata.get('author', "NA"))
          elif 'editor' in source.metadata:
            author = string_cleanup(source.metadata.get('editor', "NA"))
          reference = (
            author + " (" + 
            string_cleanup(source.metadata.get('year', "")) + "). " +
            string_cleanup(source.metadata.get('title', "")) + ". " +
            string_cleanup(source.metadata.get('address', "")) + ": " +
            string_cleanup(source.metadata.get('publisher', "")) + ".")
        elif source.metadata.get('ENTRYTYPE') == 'incollection':
          reference = (
            string_cleanup(source.metadata.get('author', "")) + " (" +
            string_cleanup(source.metadata.get('year', "")) + "). " +
            string_cleanup(source.metadata.get('title', "")) + ". " +
            "In: " +
            string_cleanup(source.metadata.get('editor', "")) + 
            " (Eds.), " +
            string_cleanup(source.metadata.get('booktitle', "")) + ", " +
            string_cleanup(source.metadata.get('pages', "")) + ".")
        else:
          author = ""
          if 'author' in source.metadata:
            author = string_cleanup(source.metadata.get('author', "NA"))
          elif 'editor' in source.metadata:
            author = string_cleanup(source.metadata.get('editor', "NA"))
          reference = (
            string_cleanup(source.metadata.get('author', "")) + " (" +
            string_cleanup(source.metadata.get('year', "")) + "). " +
            string_cleanup(source.metadata.get('title', "")) + ".")
        answer += '- '
        answer += reference
        answer += '\n'
        file.write("Query:\n")
        file.write(question)
        file.write("\n\n")
        file.write("Answer:\n")
        file.write(res['answer'])
        file.write("\n\n")
        file.write("Document: ")
        file.write(reference)
        file.write("\n")
        file.write(source.metadata['source'])
        file.write("\n\n")
        file.write("Content:\n")
        file.write(source.page_content.replace("\n", " "))
        file.write("\n\n")

  await cl.Message(content=answer).send()

 