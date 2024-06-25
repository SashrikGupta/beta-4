from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAI
import numpy as np
import faiss
import dotenv
import os

dotenv.load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS
encoder = SentenceTransformer('all-MiniLM-L6-v2')
chunks = []
index = None

# Load the PDF and create the vector space
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

    vectors = encoder.encode(chunks)
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

# Get the data based on the search query
def data_give(index, search_query):
    search_vector = encoder.encode(search_query)
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
