#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import requests  # To send requests to the Flask API

st.title("Document Query System")

# Input field for the user's query
user_input = st.text_input("Enter your query:")

# On search button press
if st.button("Search"):
    if user_input:
        # Send query to the Flask API
        response = requests.post(
            'http://127.0.0.1:5000/query',  # URL for your Flask API
            json={'query': user_input}
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            results = response.json()  # Get the JSON response from Flask
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




