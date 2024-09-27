import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from numpy import linalg

# configurar o ChromaDB com persistência local
def configure_chromadb(persist_directory):
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    return chroma_client

# gerar o a instancia de objeto para fazer o embedding
def embedding_creator():
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return embedding_model

# Função para processar PDFs e adicionar ao ChromaDB
def process_pdf_directory(pdf_directory, collection):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    print(f"Encontrados {len(pdf_files)} arquivos PDF no diretório.")
    
    for idx, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Processando PDF {idx}/{len(pdf_files)}: {pdf_file}")
        
        try:
            # Lendo o conteúdo do PDF
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            # Convertendo o texto em embedding
            embedding = embedding_creator().encode(text, convert_to_numpy = True).tolist()
            metadata = {'source': pdf_file}

            #adicionando à coleção
            collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids = [pdf_path])
            
            print(f"- Documento {pdf_file} processado e armazenado.")
        except Exception as e:
            print(f"Erro ao processar {pdf_file}: {e}")
    
    print("Processamento de todos os PDFs concluído.")

# Função para realizar consultas na coleção
def query_web_interface(collection):
    # Configurar o Streamlit
    st.set_page_config(page_title="Analisador de Currículos", page_icon="📄")
    st.title("Analisador de Currículos com RAG")

    st.write("Processamento concluído. Iniciando modo de consulta.")
    st.write("Digite sua consulta ou 'sair' para encerrar.")

    query = st.text_input("\nConsulta: ")
    if query:
        # Gerando o embedding para a consulta
        query_embedding = embedding_creator().encode(query, convert_to_numpy = True).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
        )
        st.write("\nResultados:")
        for i, (document, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), start=1):
            st.write(f"Exemplo {i}:")
            st.write(f"Fonte: {metadata['source']}")  
            st.write(document[:200]) 
            st.write("-" * 80)

# Função principal
def main():
    persist_directory = "./chroma_data"
    pdf_directory = "./pdfs"
    
    # Configurando ChromaDB com persistência local
    chroma_client = configure_chromadb(persist_directory)
    
    #criando a coleção
    collection = chroma_client.get_or_create_collection(
        name="pdfs",
    )

    # Processando o diretório de PDFs
    process_pdf_directory(pdf_directory, collection)

    # Interface para perguntas e respostas
    query_web_interface(collection)

if __name__ == "__main__":
    main()