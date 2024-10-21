# Custom Chatbot with Langchain

This project implements a custom chatbot using Langchain that extracts data from a specified URL, creates embeddings, and allows users to query the information through a Streamlit application and a Flask API.

## Project Structure


## Requirements

- Python 3.7 or higher
- Pip

## Installation

1. **Clone the repository** (replace `your-repo-url` with your actual repository URL):
    ```bash
    git clone your-repo-url
    cd your_project_directory
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Streamlit App

1. Open a terminal and navigate to your project directory.
2. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3. The app will open in your default web browser. You can enter your queries in the input field to search for information.

### Running the Flask API

1. Open another terminal and navigate to your project directory.
2. Run the Flask API:
    ```bash
    python api.py
    ```
3. The API will start on `http://127.0.0.1:5000`.

### Testing the API

You can test the Flask API using tools like **Postman** or **cURL**. Here’s an example using `curl`:

```bash
curl -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d '{"query": "your query here"}'
