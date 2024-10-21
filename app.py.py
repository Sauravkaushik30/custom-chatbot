#!/usr/bin/env python
# coding: utf-8

# In[1]:


import faiss
import numpy as np
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from transformers import AutoTokenizer, AutoModel

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

# Streamlit app
st.title("Document Query System")

# User input
user_input = st.text_input("Enter your query:")

if st.button("Search"):
    if user_input:
        # Tokenize and create embedding for user input
        user_input_tokens = tokenizer(user_input, return_tensors='pt', truncation=True, max_length=512)
        user_output = model(**user_input_tokens)
        user_embedding = user_output.last_hidden_state.mean(dim=1).detach().numpy()

        # Search in the FAISS index
        D, I = index.search(user_embedding, k=5)  # Get top 5 nearest neighbors
        responses = []

        for idx in I[0]:
            if idx >= 0 and idx < len(documents):  # Ensure index is within bounds
                responses.append(documents[idx].page_content)

        # Display results
        st.subheader("Results:")
        for i, response in enumerate(responses):
            st.write(f"{i + 1}. {response}")

    else:
        st.warning("Please enter a query.")



# In[ ]:





# In[ ]:





# In[ ]:




