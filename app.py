from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss
import dotenv
import os

dotenv.load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS
chunks = []
index = None
API_KEY=os.getenv('GOOGLE_KEY')
genai.configure(api_key=API_KEY)
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)

def embeder(text):
    sample_text = (text)

    model = 'models/embedding-001'
    embedding = genai.embed_content(model=model,
                                    content=sample_text,
                                    task_type="retrieval_document",
                                    )
    return embedding["embedding"]

def parallel_embedding(chunks):
    embeddings = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(embeder, chunk) for chunk in chunks]
        for future in as_completed(futures):
            embeddings.append(future.result())
    return embeddings

def pdf_vector_space(path_to_pdf):
    global chunks
    loader = PyPDFLoader(path_to_pdf)
    pages = loader.load()
    content = [page.page_content for page in pages]

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", " "],
        chunk_size=200,
        chunk_overlap=1
    )

    for page_content in content:
        chunks.extend(splitter.split_text(page_content))
    
    print(len(chunks))
    embeddings = parallel_embedding(chunks)
    
    # Remove None embeddings if any
    embeddings = [emb for emb in embeddings if emb is not None]
    
    vectors = np.array(embeddings)
    print(vectors)
    print(vectors.shape)
    
    if len(vectors) == 0:
        raise ValueError("No valid embeddings generated.")

    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

# Get the data based on the search query
def data_give(index, search_query):
    search_vector = embeder(search_query)
    search_vector = np.array(search_vector).reshape(1, -1)
    distance, loc = index.search(search_vector, k=60)

    data_to_be_given = ""
    for i in loc[0]:
        data_to_be_given += chunks[i]

    prompt = "you are given the following data " + data_to_be_given + "now answer the following query" + search_query
    prompt = prompt + "the answer should be highly comprehensive"
    llm  = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_KEY"))
    return llm.invoke(prompt)

@app.route('/ask', methods=['POST'])
def handle_pdf():
    global index
    if 'pdf' not in request.files:
        return "No PDF file provided", 400

    pdf = request.files['pdf']
    pdf_path = os.path.join("uploads", pdf.filename)
    pdf.save(pdf_path)

    index = pdf_vector_space(pdf_path)
    return "PDF processed and vector space created", 200

@app.route('/query', methods=['POST'])
def handle_query():
    if index is None:
        return "No PDF has been processed", 400

    data = request.get_json()
    search_query = data.get('query', '')

    if not search_query:
        return "No search query provided", 400

    result = data_give(index, search_query)
    return jsonify({"result": result})

@app.route('/delete_pdf', methods=['DELETE'])
def delete_pdf():
    global index
    if index is not None:
        index = None  # Reset the index
    pdf_path = os.path.join("uploads", request.args.get('filename', ''))
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        return "PDF file deleted successfully", 200
    else:
        return "PDF file not found", 404

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)