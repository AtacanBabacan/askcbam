import os
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

key= "yourkey"

os.environ["OPENAI_API_KEY"] = key

# Simple method - Split by pages 
loader = PyPDFLoader(r"C:\Users\Ataelif\programming\repos\github.com\askcbam\CBAM Guidance.pdf")
pages = loader.load_and_split()

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

#query = "Gecis doneminde hangi sektorler bu mekanizmaya dahil?"
query = "CBAM hangi sektorleri ilgilendiriyor?"
docs = db.similarity_search(query)

output = chain(
    {"input_documents": docs, "question": query}, return_only_outputs=True
)
print(output["output_text"])
