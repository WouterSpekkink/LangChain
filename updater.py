from langchain_community.vectorstores import Chroma
from langchain.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import bibtexparser
import langchain
import os
import glob
from dotenv import load_dotenv
import openai
import constants
import time

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Set paths
source_path = './data/src/'
destination_file = './data/ingested.txt'
store_path = './vectorstore/'
bibtex_file_path = '/home/wouter/Tools/Zotero/bibtex/library.bib'

# Load documents
print("===Loading documents===")
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(source_path,
                         show_progress=True,
                         use_multithreading=True,
                         loader_cls=TextLoader,
                         loader_kwargs=text_loader_kwargs)
documents = loader.load()

if len(documents) == 0:
    print("No new documents found")
    quit()
# Add metadata based in bibliographic information
print("===Adding metadata===")

# Read the BibTeX file
with open(bibtex_file_path) as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

# Get a list of all text file names in the directory
text_file_names = os.listdir(source_path)
metadata_store = []

# Go through each entry in the BibTeX file
for entry in bib_database.entries:
    # Check if the 'file' key exists in the entry
    if 'file' in entry:
        # Extract the file name from the 'file' field and remove the extension
        pdf_file_name = os.path.basename(entry['file']).replace('.pdf', '')

         # Check if there is a text file with the same name
        if f'{pdf_file_name}.txt' in text_file_names:
            # If a match is found, append the metadata to the list
            metadata_store.append(entry)

# Let's use filenames as unique ids
# ids = [ ]
# for document in documents:
#     name = os.path.basename(document.metadata['source']).replace('.txt', '')
#     ids.append(name) # To make our ids
            
for document in documents:
    for entry in metadata_store:
        doc_name = os.path.basename(document.metadata['source']).replace('.txt', '')
        ent_name = os.path.basename(entry['file']).replace('.pdf', '')
        if doc_name == ent_name:
            document.metadata.update(entry)

# Initialize text splitter
print("===Splitting documents into chunks===")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 150,
    length_function = len,
    add_start_index = True,
)

split_documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    show_progress_bar=True,
    request_timeout=60,
)

print("===Embedding text and updating database===")
old_db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
old_db.add_documents(split_documents)

# Record the files that we have added
print("===Recording ingested files===")
with open(destination_file, 'a') as f:
    for document in documents:
        f.write(os.path.basename(document.metadata['source']))
        f.write('\n')
