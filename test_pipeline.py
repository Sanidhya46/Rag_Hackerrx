# for testing
from loaders.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.embedder import get_embedder
from vector_store import get_vector_store 

# Load and split
docs = load_pdf("test_docs/sample.pdf")
print(f"Loaded {len(docs)} pages")

all_text = "\n".join([doc.page_content for doc in docs])
chunks = split_text(all_text)
print(f"Created {len(chunks)} chunks")

# # Show sample chunk
# print("🔹 Sample chunk:")
# print(chunks[0][:400])

# Get embedder
embedder = get_embedder()

# Generate embeddings
print("🔹 Generating embedding for the first chunk...")
embedding = embedder.embed_documents([chunks[0]])

# print("✅ Embedding shape:", len(embedding[0]))
# print("📦 First 5 values:", embedding[0][:5])

# ✅ Create and store vectors
print("🔹 Creating vector store...")
vector_store = get_vector_store(chunks)

# ✅ Run similarity search
print("🔍 Running similarity search for: 'AI chatbot'")
result = vector_store.similarity_search("words or terms mentioned below", k=1)

# ✅ Print result
print("🔹 Top result:")
print(result[0].page_content)

