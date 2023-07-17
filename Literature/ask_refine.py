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
from langchain.memory import ConversationBufferWindowMemory
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

refine_prompt_template = (
  "The original question is as follows: {question}\n"
  "We have provided an existing answer: {existing_answer}\n"
  "We have the opportunity to refine the existing answer"
  "(only if needed) with some more context below.\n"
  "------------\n"
  "{context_str}\n"
  "------------\n"
  "Given the new context, refine the original answer to better "
  "answer the question, but make sure that the answer works as a standalone answer. "
  "If the context isn't useful, return the original answer."
  "Do not refer to the original answer, as the refined answer is all the user sees. "
)

refine_prompt = PromptTemplate(
  input_variables=["question", "existing_answer", "context_str"],
  template=refine_prompt_template,
)

initial_qa_template = (
  "Use the provided piece of context below and not prior knowledge to answer the question at the end. "
  "If you don't know the answer, just say that you don't know, don't try to make up an answer, but just say 'I don't know'. "
  "Please try to give detailed answers and write your answers in academic style, unless explicitly told otherwise. "
  "---------------------\n"
  "{context_str}"
  "\n---------------------\n"
  "Answer the question: {question}\n"
)
initial_qa_prompt = PromptTemplate(
  input_variables=["context_str", "question"],
  template=initial_qa_template
)
# Set up conversational chain
chain = ConversationalRetrievalChain.from_llm(
  llm = ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=db.as_retriever(search_type="mmr", search_kwargs={"k" : 10}),
  chain_type="refine",
  return_source_documents = True,
  combine_docs_chain_kwargs={'refine_prompt': refine_prompt,
                             'question_prompt': initial_qa_prompt,
                             'return_refine_steps' : False,}
)

# Set up conversation
chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({'question': query, 'chat_history': chat_history, 'return_only_outputs' : False})
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

