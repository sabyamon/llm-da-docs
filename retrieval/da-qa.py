# Import necessary libraries
import os
from langchain.chains import RetrievalQA  # For creating QA chains
from langchain_openai import OpenAIEmbeddings  # For generating embeddings using OpenAI
from langchain_community.vectorstores import Pinecone as LangchainPinecone  # Vector store integration
from pinecone import Pinecone, ServerlessSpec  # Pinecone vector database
from openai import OpenAI  # OpenAI API client

# Step 1: Initialize OpenAI and embeddings
# Set up API key for OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = OpenAI(
  api_key=openai_api_key,
)
# Set up embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
embed_model = "text-embedding-3-small"

# Step 2: Initialize Pinecone
# Set up Pinecone credentials and configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("INDEX_NAME")

# Initialize Pinecone client and connect to index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Step 3: Create embeddings for the query
# Define the query
query = "How do I search in the DA?"
# Generate embeddings for the query using OpenAI
res = client.embeddings.create(
    input=[query],
    model=embed_model
)
# Extract the embedding vector
query_vector = res.data[0].embedding

# Step 4: Search in Pinecone
# Query Pinecone index with the embedding vector
retrieval_results = index.query(
    vector=query_vector,
    top_k=3,  # Get top 3 most similar documents
    include_metadata=True
)

# Step 5: Process retrieved documents
# Combine retrieved documents into a single context string
context = ""
for match in retrieval_results["matches"]:
    context += f"{match['metadata']['title']}\nURL: {match['metadata']['url']}\nContent: {match['metadata']['content']}\n\n"

# Step 6: Generate answer using GPT
# Create prompt combining context and query
prompt = f"""
You are a helpful assistant for developer documentation. Use the following context to answer the question. Include relevant documentation URLs in your response:

Context:
{context}

Question: {query}

Answer:
"""

# Call OpenAI API to generate response
response = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL"),
    messages=[{"role": "user", "content": prompt}],
)

# Step 7: Extract and print the answer
for choice in response.choices:
    resp = choice.message.content.strip()
    print("Answer:", resp)

print(response)