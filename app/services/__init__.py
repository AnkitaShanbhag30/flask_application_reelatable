import os
from pinecone import Pinecone

pinecone_api_key = os.getenv('PINECONE_API_KEY')
hf_token = os.getenv('HF_TOKEN')

INDEX_NAME = 'reelatable-embeddings'
INDEX_DIMENSION = 1024

if not pinecone_api_key:
    raise ValueError("No Pinecone API key found in environment variables")

pinecone_client = Pinecone(api_key=pinecone_api_key)
index = pinecone_client.Index(name=INDEX_NAME)