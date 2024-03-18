from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_community.document_transformers import LongContextReorder, EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from datetime import datetime
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
import os
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

# Load Chroma database
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)

# Set up callback handler
handler = OpenAICallbackHandler()

# Set memory
memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True, k = 3)

# Customize prompt
system_prompt_template = (
  '''
  You are a knowledgeable professor working in academia.
  Using the provided pieces of context, you answer the questions asked by the user.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.

  """
  Context: {context}
  """

  Please try to give detailed answers and write your answers as an academic text, unless explicitly told otherwise.
  Use references to literature in your answer and include a bibliography for citations that you use.
  If you cannot provide appropriate references, tell me by the end of your answer.
 
  Format your answer as follows:
  One or multiple sentences that constitutes part of your answer (APA-style reference)
  The rest of your answer
  Bibliography:
  Bulleted bibliographical entries in APA-style
  ''')
  
system_prompt = PromptTemplate(template=system_prompt_template,
                               input_variables=["context"])

system_message_prompt = SystemMessagePromptTemplate(prompt = system_prompt)
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, reordering])

metadata_field_info = [
    AttributeInfo(
        name="title",
        description="The title of the document",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="""
        The author or authors of the document. The names are spelt with the last name first: [LAST NAME], [OPTIONAL MIDDLE INITIALS], [FIRST NAMES].
        Queries by the user will rarely be an exact match with metadata recorded in this field.
        """, 
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year that the document was published",
        type="integer",
    ),
    AttributeInfo(
        name="journal",
        description="The name of the journal in which the document was published.",
        type="string",
    ),
]
document_content_description = "A paragraph from an academic publication"
sq_retriever = SelfQueryRetriever.from_llm(
  llm=ChatOpenAI(model="gpt-4-0125-preview"),
  vectorstore=db,
  document_contents=document_content_description,
  metadata_field_info=metadata_field_info,
  search_kwargs={"k":30}
)

retriever = ContextualCompressionRetriever(
#  base_compressor=pipeline, base_retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k" : 20, "score_threshold": .5}))
  #base_compressor=pipeline, base_retriever=db.as_retriever(search_type="similarity", search_kwargs={"k" : 20}))
  base_compressor=pipeline, base_retriever=sq_retriever)
 
# Set up source file
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"answers/answers_{timestamp}.org"
with open(filename, 'w') as file:
  file.write("#+OPTIONS: toc:nil author:nil\n")
  file.write(f"#+TITLE: Answers and sources for session started on {timestamp}\n\n")

# Prepare settings
@cl.on_chat_start
async def start():
  settings = await cl.ChatSettings(
    [
      Select(
        id="Model",
        label="OpenAI - Model",
        values=["gpt-3.5-turbo-16k", "gpt-4-1106-preview", "gpt-4-0125-preview"],
        initial_index=2,
      ),
      Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
      Slider(
        id="Temperature",
        label="OpenAI - Temperature",
        initial=0,
        min=0,
        max=2,
        step=0.1,
      ),
    ]
  ).send()
  await setup_chain(settings)

# When settings are updated
@cl.on_settings_update
async def setup_chain(settings):
  # Set llm
  llm=ChatOpenAI(
    temperature=settings["Temperature"],
    streaming=settings["Streaming"],
    model=settings["Model"],
  )
 
  # Set up conversational chain
  chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents = True,
    return_generated_question = True,
    combine_docs_chain_kwargs={'prompt': chat_prompt},
    memory=memory,
    condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
  )
  cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: str):
  chain = cl.user_session.get("chain")
  cb = cl.LangchainCallbackHandler()
  cb.answer_reached = True
  res = await cl.make_async(chain)(message.content, callbacks=[cb])
  question = res["question"]
  answer = res["answer"]
  answer += "\n\n # Sources:\n\n"
  sources = res["source_documents"]
  counter = 1
  for source in sources:
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
    answer += (f"## Document_{counter}:\n- ")
    answer += (reference)
    answer += ("\n- ")
    answer += (os.path.basename(source.metadata['source']))
    answer += ("\n")
    answer += ("### Content:\n")
    answer += (source.page_content)
    answer += ("\n\n")
    counter += 1

  await cl.Message(content=answer).send()
