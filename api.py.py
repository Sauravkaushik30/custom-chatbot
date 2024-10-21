#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    # Process the query like in the Streamlit app, return results
    ...
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

