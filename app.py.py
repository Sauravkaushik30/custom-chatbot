#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter


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


st.title("Document Query System")


user_input = st.text_input("Enter your query:")


if st.button("Search"):
    if user_input:
     
        user_input_tokens = tokenizer(user_input, return_tensors='pt', truncation=True, max_length=512)
        user_output = model(**user_input_tokens)
        user_embedding = user_output.last_hidden_state.mean(dim=1).detach().numpy()

        D, I = index.search(user_embedding, k=5) 
        responses = []

        for idx in I[0]:
            if idx >= 0 and idx < len(chunks):  
                responses.append(chunks[idx]) 

        
        st.subheader("Results:")
        for i, response in enumerate(responses):
            st.write(f"{i + 1}. {response}")
    else:
        st.warning("Please enter a query.")


# In[ ]:





# In[ ]:





# In[ ]:




