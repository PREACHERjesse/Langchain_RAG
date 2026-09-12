from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from filereader import read_multiple_pdfs
from langchain_huggingface import HuggingFaceEmbeddings




embeddings = HuggingFaceEmbeddings(
    model_name="moka-ai/m3e-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = InMemoryVectorStore(embeddings)

DOC_PATHS = [
    "pdfs/climate_change (1).pdf",
    "pdfs/global_warming.pdf"
]

docs = read_multiple_pdfs(DOC_PATHS)

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

all_splits = text_spliter.split_documents(docs)
print(f"split dpcummentation into {len(all_splits)} chunks")

vector_store.add_documents(documents=all_splits)
print(f"indexed {len(all_splits)} chunks")
