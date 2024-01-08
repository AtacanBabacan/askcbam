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

key = "yourkey"
os.environ["OPENAI_API_KEY"] = key

# You MUST add your PDF to local files in this notebook (folder icon on left hand side of screen)

# Advanced method - Split by chunk

# Step 1: Convert PDF to text
reader = PdfReader(r"C:\Users\Ataelif\programming\repos\github.com\askcbam\CBAM Guidance.pdf") 

# extracting text from each page 
doc = ""
for page in reader.pages:
    doc += page.extract_text() + "\n"

# Step 2: Save to .txt and reopen (helps prevent issues)
with open(r"C:\Users\Ataelif\programming\repos\github.com\askcbam\CBAM Guidance.txt", "w", encoding="utf-8") as f:
    f.write(doc)

with open("CBAM Guidance.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
encoded = tokenizer(text, truncation=True, max_length=1024)
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
type(chunks[0]) 

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



