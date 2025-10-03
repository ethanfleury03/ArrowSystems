from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

HF_CACHE = "C:/Users/ethan/.cache/huggingface/hub"

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=HF_CACHE
)

def build_index(data_dir="data", storage_dir="storage"):
    if os.path.exists(storage_dir):
        print("🔄 Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context)

    print("📥 Creating new index from data...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir=storage_dir)
    print("✅ Index created and cached.")
    return index

if __name__ == "__main__":
    build_index()
