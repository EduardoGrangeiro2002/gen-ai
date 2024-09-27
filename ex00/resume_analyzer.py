from datetime import datetime
import chromadb
import os
import PyPDF2
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

def read_pdf(path):
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def process_pdf_directory(pdf_directory, collection):
    print("Encontrados 40 arquivos PDF no diretório.")
    i = 0
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            i += 1
            print(f"Processando PDF {i}/50: {filename}")
            file_path = os.path.join(pdf_directory, filename)
            text = read_pdf(file_path)
            collection.add(
                documents=[text],
                embeddings=[embbedings(text)],
                ids=[get_timestamp()],
                metadatas=[{"source": filename}]
            )
            print(f"- Documento {filename} processado e armazenado.")


def interactive_query_loop(collection):
    while True:
        query = input("\nConsulta: ")
        if query.lower() == 'sair':
            break
            
        results = collection.query(
            query_embeddings=[embbedings(query)],
            n_results=5
        )
        print("\nResultados:")
        for i, (document, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), start=1):
            if (document != None):
                print(f"Exemplo {i}:")
                print(f"Fonte: {metadata['source']}")
                print(document[:200]) 
                print("-" * 80)

def get_timestamp():
    return str(int(datetime.now().timestamp() * 1000))

def embbedings(text):
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embedding = embedding_model.encode(text, convert_to_numpy = True).tolist()
    return embedding

def main():
    persist_directory = "./chroma_data"
    pdf_directory = "./curriculos"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_or_create_collection(name="Curriculum")
    collection.delete()
    process_pdf_directory(pdf_directory, collection)
    interactive_query_loop(collection)

if __name__ == "__main__":
    main()