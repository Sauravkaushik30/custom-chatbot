#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')


url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc.page_content))


dimension = 768 
index = faiss.IndexFlatL2(dimension)


embeddings = []
for chunk in chunks:
    inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    embeddings.append(embedding)

embeddings = np.vstack(embeddings)


index.add(embeddings)

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

  
    user_input_tokens = tokenizer(user_query, return_tensors='pt', truncation=True, max_length=512)
    user_output = model(**user_input_tokens)
    user_embedding = user_output.last_hidden_state.mean(dim=1).detach().numpy()

  
    D, I = index.search(user_embedding, k=5)  
    responses = []

    for idx in I[0]:
        if idx >= 0 and idx < len(chunks): 
            responses.append(chunks[idx])  

    
    return jsonify({"results": responses})

if __name__ == '__main__':
    app.run(debug=True)


