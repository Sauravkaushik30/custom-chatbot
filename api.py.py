#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import faiss
import numpy as np
from langchain.document_loaders import WebBaseLoader
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

# Load data from the URL
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()

# Initialize FAISS index
dimension = 768  # DistilBERT output size
index = faiss.IndexFlatL2(dimension)

# Create embeddings for all documents
embeddings = []
for doc in documents:
    inputs = tokenizer(doc.page_content, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    embeddings.append(embedding)

# Convert list of embeddings to a numpy array
embeddings = np.vstack(embeddings)

# Add embeddings to the FAISS index
index.add(embeddings)

# Flask route to handle user queries
@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')

    # Create embedding for user input
    user_input_tokens = tokenizer(user_query, return_tensors='pt', truncation=True, max_length=512)
    user_output = model(**user_input_tokens)
    user_embedding = user_output.last_hidden_state.mean(dim=1).detach().numpy()

    # Search in the FAISS index
    D, I = index.search(user_embedding, k=5)  # Get top 5 nearest neighbors
    responses = []

    for idx in I[0]:
        if idx >= 0 and idx < len(documents):  # Ensure index is within bounds
            responses.append(documents[idx].page_content)

    return jsonify(responses)

if __name__ == '__main__':
    app.run(debug=True)

