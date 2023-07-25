# Introduction
This repo contains my current langchain projects.
The main project at the moment focuses on augmented retrieval with academic publications in my Zotero library. 
This project makes use of several python scripts:
- A script to make a new index when starting from scratch (indexer.py)
- A script to update an existing index after new papers have been added to the library (updater.py)
- A script to remove corrupt documents from the index (remove_docs.py)
- A script that does the actual Q&A, using GPT3.5-turbo as the main LLM (ask.py)
- I have a second version of the last script that uses chainlit for a nicer user interface (ask_cl.py).

There is a second project in which I experiment with open source LLMs.
It is contained in a folder called HuggingFace, but I actually don't focus exclusively on HugginFace based solutions here.
