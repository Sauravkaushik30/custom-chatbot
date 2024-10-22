#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import faiss
import numpy as np
from langchain.document_loaders import WebBaseLoader
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')


url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()


dimension = 768  
index = faiss.IndexFlatL2(dimension)


embeddings = []
for doc in documents:
    inputs = tokenizer(doc.page_content, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    embeddings.append(embedding)

embeddings = np.vstack(embeddings)

index.add(embeddings)

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')


    user_input_tokens = tokenizer(user_query, return_tensors='pt', truncation=True, max_length=512)
    user_output = model(**user_input_tokens)
    user_embedding = user_output.last_hidden_state.mean(dim=1).detach().numpy()

  
    D, I = index.search(user_embedding, k=5) 
    responses = []

    for idx in I[0]:
        if idx >= 0 and idx < len(documents):  
            responses.append(documents[idx].page_content)

    return jsonify(responses)

if __name__ == '__main__':
    app.run(debug=True)

