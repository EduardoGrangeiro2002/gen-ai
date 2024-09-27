from functools import wraps
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


import chromadb
import os
import uuid


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './curriculums'

# Simulated database (in-memory)
curriculums_db = {}
users_db = {
    "candidate": "candidate",
    "recruiter": "recruiter",
    "admin": "admin"
}


def chunk_text_recursive(text, chunk_size=1000, chunk_overlap=200):
    # Inicializa o splitter com o tamanho do chunk e sobreposição especificada
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    # Divide o texto em chunks
    chunks = splitter.split_text(text)
    return chunks

def configure_chromadb(persist_directory):
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    return chroma_client

chroma_client = configure_chromadb("./chroma_data")


def embedding_creator():
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return embedding_model

# Helper: Simple Role Checker
def check_role(role_required):
    def decorator(f):
        @wraps(f) 
        def wrapper(*args, **kwargs):
            username = request.headers.get('username')
            if username not in users_db:
                return jsonify({'message': 'User not found!'}), 403
            user_role = users_db[username]
            if user_role != role_required and user_role != 'admin':
                return jsonify({'message': 'Permission denied!'}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator

# Endpoint: Upload PDF
@app.route('/upload_pdf', methods=['POST'])
@check_role('candidate')  # Only candidates or admin can upload
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    
    print(file)

    filename = file.filename

    curriculum_id = str(uuid.uuid4())

    collection = chroma_client.get_or_create_collection(
        name="pdfs",
    )


    try:
        # Lendo o conteúdo do PDF
        
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Convertendo o texto em embedding
        chunks = chunk_text_recursive(text)
        chunk_id = 0
        metadatas={'source': filename, 'source_chunk':str(chunk_id), 'type': request.form['type']}
        for text in chunks:
            embbeded_chunk = embedding_creator().encode(text, convert_to_numpy = True).tolist()
            collection.add(
            documents=[text],
            embeddings=[embbeded_chunk],
            metadatas=[metadatas],
            ids = [filename])
            chunk_id += 1
        

        #embedding = embedding_creator().encode(text, convert_to_numpy = True).tolist()

        
        
    except Exception as e:
        print(f"Erro ao processar documento: {e}")

    return jsonify({'message': f'File uploaded successfully! {filename}'})


# Endpoint: Search for Curriculums
@app.route('/search', methods=['GET'])
def search():

    username = request.headers.get('username')
    user_role = users_db.get(username, '')

    collection = chroma_client.get_or_create_collection(
        name="pdfs",
    )

    query_embedding = embedding_creator().encode(request.args.get("query"), convert_to_numpy = True).tolist()

    if user_role == 'candidate':
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
    elif user_role in ['recruiter', 'admin']:
        #query para pesquisar nos currilos
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
    else:
        return jsonify({'message': 'Permission denied!'}), 403

    return jsonify({'curriculums': results["documents"][0]})

# Endpoint: Delete a Curriculum (Only Admin)
@app.route('/curriculum/<filename>', methods=['DELETE'])
@check_role('admin')
def delete_curriculum(filename):

    collection = chroma_client.get_or_create_collection(
        name="pdfs",
    )
    try:
        collection.delete(
        where={"source": filename}
        )
        return jsonify({'message': 'Curriculum deleted successfully'})
    except:
        return jsonify({'message': 'Error at deletion'})

if __name__ == '__main__':
    app.run(debug=True)