Here’s the finalized and combined `README.md` file:

# Custom Chatbot using Langchain and FAISS

## Overview
This project demonstrates how to build a custom chatbot that retrieves relevant chunks of data from a URL using Langchain's URL loaders and FAISS indexing for fast retrieval. It provides two interfaces:
- **Streamlit Web App**: A user-friendly interface for querying documents.
- **Flask REST API**: A programmatic API to handle user queries.

## Features
1. **Data Extraction**: Extracts data from the URL [Brainlox Technical Courses](https://brainlox.com/courses/category/technical) using Langchain's `WebBaseLoader`.
2. **Document Chunking**: Splits large documents into smaller chunks for fine-grained embedding and search.
3. **Embeddings**: Generates embeddings for each chunk using `DistilBERT` and stores them in a FAISS index.
4. **Streamlit App**: Provides an interactive web interface for users to query relevant data.
5. **Flask API**: Exposes a REST API for programmatic querying of document chunks.

## Setup Instructions

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to isolate dependencies and avoid conflicts with other projects.

- For Windows:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- For Mac/Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
After activating the virtual environment, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
To run the interactive search app using Streamlit:

```bash
streamlit run app.py.py
```

This will launch the Streamlit app on your browser, allowing you to enter queries and receive relevant chunks of data.

### 5. Run the Flask REST API
To run the Flask REST API:

```bash
python api.py.py
```

Once the Flask app is running, you can send POST requests to `http://127.0.0.1:5000/query` with your search queries.

For example, using `curl`:

```bash
curl -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d '{"query": "Java"}'
```

This will return the most relevant chunks from the documents based on the provided query.

### 6. Accessing the Application

- **Streamlit Web App**: Available at `http://localhost:8501` after running `streamlit run app.py`.
- **Flask API**: Send requests to `http://127.0.0.1:5000/query` for query-based document search.

## File Structure

- `app.py.py`: The Streamlit-based user interface for querying.
- `api.py.py`: Flask REST API for handling user queries and returning document chunks.
- `requirements.txt`: Contains the required Python dependencies.
- `README.md`: Project documentation.

## Usage

1. **Streamlit Web App**: After running the app, input your query and get document chunks that are the closest match to your query.
2. **Flask API**: Use tools like `Postman` or `curl` to send POST requests to the API with the query in the request body. The API will return the top 5 relevant chunks.

## Dependencies
- `faiss-cpu`: FAISS for efficient similarity search.
- `transformers`: For tokenization and generating embeddings.
- `langchain`: To load and process documents.
- `streamlit`: For creating the web-based user interface.
- `Flask`: For building the REST API.

### Ensure these dependencies are installed via `requirements.txt` inside your virtual environment:
```bash
pip install -r requirements.txt
```

## Example Query

Using the Flask API:
```bash
curl -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d '{"query": "Java"}'
```

This will return chunks related to "Java" from the extracted documents.
