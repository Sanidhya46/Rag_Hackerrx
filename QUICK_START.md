# 🚀 **RAG SYSTEM - QUICK START GUIDE**

## 📋 **What You'll Get**
A **production-ready** RAG (Retrieval-Augmented Generation) system that can:
- ✅ Handle **any type of documents** (PDF, Word, Excel, etc.)
- ✅ Provide **accurate, context-aware answers**
- ✅ Work **fast and reliably** without crashes
- ✅ Scale to **production environments**
- ✅ Use **GPU acceleration** when available

## 🎯 **System Architecture**
```
User Question → Text Embedding → Similarity Search → Context Retrieval → AI Answer Generation
```

## ⚡ **Quick Installation (3 Steps)**

### **Step 1: Run the Auto-Installer**
```bash
# This will do everything automatically
python install_backend.py
```

### **Step 2: Start the Server**
```bash
# Windows
start_backend.bat

# Linux/Mac
./start_backend.sh

# Or manually
.\env\Scripts\activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **Step 3: Test the System**
- 🌐 **API Docs**: http://localhost:8000/docs
- 🏥 **Health Check**: http://localhost:8000/health
- 📤 **Upload Document**: Use the `/upload` endpoint
- ❓ **Ask Questions**: Use the `/query` endpoint

## 🔧 **Manual Installation (If Auto-Installer Fails)**

### **1. Create Virtual Environment**
```bash
python -m venv env

# Activate (Windows)
.\env\Scripts\Activate.ps1

# Activate (Linux/Mac)
source env/bin/activate
```

### **2. Install Dependencies**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all packages
pip install -r requirements.txt
```

### **3. Verify Installation**
```bash
python -c "import fastapi, torch, sentence_transformers; print('✅ All packages installed!')"
```

## 📦 **What Gets Installed**

### **Core AI Components**
- **Sentence Transformers**: Converts text to numerical vectors
- **PyTorch**: Deep learning framework for model inference
- **NumPy**: Fast numerical operations for vectors

### **Document Processing**
- **PyPDF2 + pdfplumber**: Advanced PDF text extraction
- **python-docx**: Microsoft Word document support
- **openpyxl**: Excel spreadsheet support
- **pytesseract**: Extract text from images (OCR)

### **Web Framework**
- **FastAPI**: Modern, fast web framework
- **Uvicorn**: Lightning-fast ASGI server

## 🎮 **How to Use the System**

### **1. Upload a Document**
```bash
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

### **2. Ask a Question**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the waiting period for maternity coverage?"}'
```

### **3. Get Context-Aware Answers**
The system will:
1. Convert your question to a numerical vector
2. Find the most relevant document chunks
3. Generate an accurate answer using the retrieved context

## 🚨 **Troubleshooting Common Issues**

### **Import Errors**
```bash
# Make sure virtual environment is activated
.\env\Scripts\Activate.ps1  # Windows
source env/bin/activate     # Linux/Mac
```

### **Model Download Issues**
```bash
# Models download automatically on first use
# If they fail, check internet connection and try again
```

### **GPU Not Working**
```bash
# Check if CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Install PyTorch with CUDA support if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Port Already in Use**
```bash
# Change port in the startup command
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## 🔒 **Production Deployment**

### **Security Checklist**
- [ ] Update `.env` file with secure keys
- [ ] Set `DEBUG=False`
- [ ] Use HTTPS in production
- [ ] Implement proper authentication
- [ ] Set up monitoring and logging

### **Performance Optimization**
- [ ] Use GPU acceleration (CUDA/MPS)
- [ ] Implement caching for embeddings
- [ ] Use production-grade ASGI server
- [ ] Set up load balancing if needed

## 📚 **Advanced Features**

### **Custom Models**
```python
# Use different embedding models
from utils.embedder import DocumentEmbedder

embedder = DocumentEmbedder(
    model_name="all-mpnet-base-v2",  # Better quality, slower
    device="cuda"  # Force GPU usage
)
```

### **Batch Processing**
```python
# Process multiple documents efficiently
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = embedder.encode_batch(texts, batch_size=32)
```

### **Similarity Search**
```python
# Find most similar documents
query = "What are the policy terms?"
query_embedding = embedder.encode_text(query)
similar_docs = embedder.find_most_similar(query_embedding, doc_embeddings, top_k=5)
```

## 🆘 **Need Help?**

### **Check Logs**
```bash
# Look for error messages in the console output
# The system provides detailed logging for debugging
```

### **Common Error Messages**
- **"Model not initialized"**: Run the auto-installer again
- **"CUDA out of memory"**: Reduce batch size or use CPU
- **"Port already in use"**: Change port number
- **"Import error"**: Activate virtual environment

### **Performance Tips**
- **GPU**: Use NVIDIA GPU for 10x faster processing
- **Batch Size**: Optimal batch size is 32 for most systems
- **Memory**: Ensure you have at least 4GB RAM available
- **Storage**: Models are cached locally (~100MB per model)

## 🎉 **You're Ready!**

Your RAG system is now:
- ✅ **Fast**: GPU-accelerated processing
- ✅ **Reliable**: Fallback models and error handling
- ✅ **Scalable**: Production-ready architecture
- ✅ **Accurate**: Context-aware answer generation

**Start asking questions and get AI-powered answers from your documents!**

