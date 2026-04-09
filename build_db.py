from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Loading PDFs")
loader = PyPDFDirectoryLoader("./skin_disease_pdfs")
documents = loader.load()
print(f"Loaded {len(documents)} pages.")

print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks.")

print("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Building ChromaDB vector database...")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model, 
    persist_directory="./chroma_db" 
)
print("Database built successfully!")