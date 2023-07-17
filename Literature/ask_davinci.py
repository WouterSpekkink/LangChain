from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
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
prompt_template = ("Use the provided pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. "
"Please try to give detailed answers and write your answers in academic language, unless explicitly told otherwise. "
"------------\n"
"Context: {context}\n"
"------------\n"
"Question: {question}")

PROMPT = PromptTemplate(
  template=prompt_template,
  input_variables=["context", "question"]
)
prompt = PROMPT

qa = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(model='text-davinci-003', max_tokens=1000),
                                          question_prompt=prompt,
                                          retriever=db.as_retriever(search_type="mmr", search_kwargs={"k" : 10}),
)
result = qa(query)
question = result['question']
answer = result['answer']
sources = result['sources']
print("Question:\n", question)
print("\n")
print("Answer:\n", answer)
print("\n")
print("Sources:\n")
for source in sources:
  source = os.path.basename(source)
print(sources)



