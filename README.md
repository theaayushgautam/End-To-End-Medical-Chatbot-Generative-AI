# End-to-end-Medical-Chatbot-Generative-AI 🤖


# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone

### Code Explaination

This code is the backend for your Medibot project, and it sets up a Flask-based web application that handles user interactions and queries, processes them through an AI-based pipeline (using LangChain, Pinecone, OpenAI's API, and Hugging Face embeddings), and retrieves answers based on a large language model (LLM) like GPT. Here's a step-by-step explanation of the code:

1. Imports
python
Copy code
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

Flask: Web framework for handling HTTP requests and responses.
download_hugging_face_embeddings: Custom function (from src.helper) for downloading pre-trained embeddings from Hugging Face.

PineconeVectorStore: LangChain's interface to store and retrieve vector embeddings using Pinecone.

OpenAI: Wrapper for OpenAI’s GPT API (LLM).
create_retrieval_chain: LangChain’s function to create a retrieval-based AI chain, combining information retrieval and language generation.

dotenv: Loads environment variables from a .env file (e.g., API keys).

os: Handles environment variable management.

2. Initialize Flask App
python
Copy code
app = Flask(__name__)
app: This initializes the Flask application that will serve as the backend of the Medibot project.

3. Load Environment Variables
python
Copy code
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
load_dotenv(): Loads the environment variables (like API keys) from the .env file.
os.environ.get(): Retrieves the Pinecone and OpenAI API keys from the environment variables.
These variables are then set in the environment (os.environ), making them accessible to the API clients.

4. Download Hugging Face Embeddings
python
Copy code
embeddings = download_hugging_face_embeddings()
This function (from src.helper) downloads and sets up Hugging Face embeddings, which will later be used to convert text chunks (documents) into vector embeddings for semantic search.

5. Pinecone VectorStore Setup
python
Copy code
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
PineconeVectorStore.from_existing_index: Connects to the Pinecone vector database (which stores precomputed embeddings) using the index name medicalbot.
index_name: The name of the Pinecone index where all the document embeddings are stored.
docsearch: This acts as a retriever for querying similar documents from Pinecone's index based on vector embeddings.

6. Retriever Setup
python
Copy code
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
docsearch.as_retriever(): Converts the Pinecone vector store into a retriever that can perform similarity searches.
search_type="similarity": Specifies that the retriever should use similarity-based search.
search_kwargs={"k":3}: When a query is made, it retrieves the top 3 similar results from the Pinecone index.

7. Setup OpenAI GPT Model
python
Copy code
llm = OpenAI(temperature=0.4, max_tokens=500)
OpenAI(): Initializes the OpenAI GPT model with specific parameters.
temperature=0.4: Controls the creativity of the model's responses (lower values make the output more deterministic).
max_tokens=500: Limits the length of the generated response to 500 tokens.

8. Define the Prompt Template
python
Copy code
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
ChatPromptTemplate.from_messages: Defines a template for how the GPT model should process inputs.
system_prompt: This would be a predefined system-level instruction to guide the model's behavior (e.g., "Act like a medical assistant").
human: The actual input provided by the user ("{input}" is the placeholder for user input).

9. Create Chains (Combining Document Retrieval with LLM)
python
Copy code
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
create_stuff_documents_chain: Creates a document processing chain that can take retrieved documents and "stuff" them into a prompt for the LLM.
create_retrieval_chain: Combines the retriever (document search) and the question-answering chain, creating a full AI pipeline.
The retrieval chain uses the embeddings to find relevant document chunks and passes them to the LLM to generate a response.

10. Routes Setup
Home Route:
python
Copy code
@app.route("/")
def index():
    return render_template('chat.html')
@app.route("/"): Defines the root endpoint (home page). When users visit this URL, it renders an HTML page (chat.html) that likely contains a chat interface.
Chat Route (Handling User Queries):
python
Copy code
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])
@app.route("/get", methods=["GET", "POST"]): Defines a route that can handle both GET and POST requests. This route processes the user query.
msg = request.form["msg"]: Retrieves the user's message from the form (front-end input).
rag_chain.invoke({"input": msg}): Sends the user's query to the rag_chain, which first retrieves relevant documents and then passes them to the LLM for a response.
response["answer"]: Extracts the generated answer from the model's output.
return str(response["answer"]): Returns the generated answer back to the user as a string (likely displayed in the chat UI).

11. Running the Application

python
Copy code
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
app.run(): Runs the Flask web server.
host="0.0.0.0": The server will be accessible externally (on any IP).
port=5000: Specifies that the app will run on port 5000.
debug=True: Enables Flask's debugging mode, which provides detailed error messages and reloads the server on changes.


    

