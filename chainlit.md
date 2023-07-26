# Welcome to Wouter's literature chatbot
This chatbot makes use of augmented retrieval to ask questions about the literature included in Wouter's Zotero Library.
The literature is stored in a FAISS index. 
Up to 10 documents are fetched for each query.
These are found through the maximal marginal relevance search algorithm, to balance between diversity of the documents and the semantic similarity of their contents to the query of the suser.

Please write your question below. You will get a response, including the sources used in the response.
A detailed log of the chat is written to a timestamped text file in the 'answers' folder.
The log not only keeps track of questions, answers and sources, but also records the relevant fragments of text from the sources.

