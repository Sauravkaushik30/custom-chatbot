#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import requests  

st.title("Document Query System")


user_input = st.text_input("Enter your query:")


if st.button("Search"):
    if user_input:
        
        response = requests.post(
            'http://127.0.0.1:5000/query', 
            json={'query': user_input}
        )
        
  
        if response.status_code == 200:
            results = response.json()  
            st.subheader("Results:")
            for i, result in enumerate(results):
                st.write(f"{i + 1}. {result}")
        else:
            st.error("Failed to retrieve results from the API.")
    else:
        st.warning("Please enter a query.")



# In[ ]:





# In[ ]:





# In[ ]:




