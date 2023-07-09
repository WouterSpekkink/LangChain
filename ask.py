from annoy import AnnoyIndex
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Load the Annoy index
dimension = 512
index = AnnoyIndex(dimension, metric='euclidean')
index.load("./vectorstore/index.ann")

# Get query as argument
query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

# TODO: I MIGHT WANT TO FIRST SELECT THE MOST RELEVANT INFORMATION
# I THINK THIS WILL LIMIT THE SCOPE OF ANY FURTHER CONVERSATION
relevant_documents = index.similarity_search(query)
retrieved_information = retrieve_information(relevant_documents)

# Set up elements for chain
llm_model = "gpt3-turbo"
retriever = index
chain = ConversationalRetrievalChain.from_llm(llm_model, retriever)

# THE BELOW NEEDS TO BE REPLACED WITH SOMETHING LIKE THE ABOVE
  
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
