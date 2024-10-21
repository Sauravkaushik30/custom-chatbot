import faiss
import numpy as np
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from transformers import AutoTokenizer, AutoModel

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
            if idx >= 0 and idx < len(documents):  
                responses.append(documents[idx].page_content)

       
        st.subheader("Results:")
        for i, response in enumerate(responses):
            st.write(f"{i + 1}. {response}")

    else:
        st.warning("Please enter a query.")









# In[ ]:





# In[ ]:




