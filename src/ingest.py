# src/ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "./document.pdf")

# DB
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE", "vector_db")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")
COLLECTION = os.getenv("PGVECTOR_COLLECTION", "documents")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Providers
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()

def _get_embeddings():
    if EMBEDDINGS_PROVIDER == "gemini":
        # models/embedding-001
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # default: OpenAI text-embedding-3-small
    return OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_pdf():
    # 1) Carrega PDF
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # 2) Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)

    # 3) Embeddings
    embeddings = _get_embeddings()

    # 4) Vetor Store (PGVector)
    conn_str = f"postgresql+psycopg://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=conn_str,
        use_jsonb=True,  # guarda metadados em jsonb
        pre_delete_collection=False,
    )

    # 5) Persistência
    vectorstore.add_documents(chunks)
    print(f"Ingestão concluída: {len(chunks)} chunks gravados em '{COLLECTION}'.")

if __name__ == "__main__":
    ingest_pdf()
