#!/usr/bin/env python3
"""
===============================================================================
COMPLETE DEPENDENCY INSTALLATION SCRIPT FOR RAG SYSTEM
===============================================================================

PURPOSE:
This script installs all required dependencies for the production-ready RAG system
and fixes common installation issues.

FEATURES:
- Automatic dependency installation
- Environment setup
- Common issue fixes
- Verification of installation

USAGE:
python install_dependencies.py
===============================================================================
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"\n🔄 {description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("env")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("🔄 Creating virtual environment...")
    if run_command("python -m venv env", "Creating virtual environment"):
        print("✅ Virtual environment created successfully")
        return True
    return False

def activate_virtual_environment():
    """Activate virtual environment based on OS"""
    system = platform.system().lower()
    
    if system == "windows":
        activate_script = "env\\Scripts\\activate"
        pip_path = "env\\Scripts\\pip"
        python_path = "env\\Scripts\\python"
    else:
        activate_script = "source env/bin/activate"
        pip_path = "env/bin/pip"
        python_path = "env/bin/python"
    
    print(f"🔄 Activating virtual environment...")
    print(f"Use: {activate_script}")
    
    return pip_path, python_path

def upgrade_pip(pip_path):
    """Upgrade pip to latest version"""
    return run_command(f"{pip_path} install --upgrade pip", "Upgrading pip")

def install_core_dependencies(pip_path):
    """Install core dependencies"""
    core_packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "python-dotenv==1.0.0",
        "structlog==23.2.0",
        "python-json-logger==2.0.7"
    ]
    
    for package in core_packages:
        if not run_command(f"{pip_path} install {package}", f"Installing {package}"):
            return False
    return True

def install_google_gemini_dependencies(pip_path):
    """Install Google Gemini dependencies"""
    gemini_packages = [
        "google-generativeai==0.8.3",
        "google-ai-generativelanguage==0.6.0",
        "google-auth==2.28.1",
        "google-auth-oauthlib==1.2.0",
        "google-auth-httplib2==0.2.0"
    ]
    
    for package in gemini_packages:
        if not run_command(f"{pip_path} install {package}", f"Installing {package}"):
            return False
    return True

def install_document_processing_dependencies(pip_path):
    """Install document processing dependencies"""
    doc_packages = [
        "PyPDF2==3.0.1",
        "pdfplumber==0.10.3",
        "python-docx==1.1.0",
        "openpyxl==3.1.2",
        "Pillow==10.1.0",
        "pytesseract==0.3.10"
    ]
    
    for package in doc_packages:
        if not run_command(f"{pip_path} install {package}", f"Installing {package}"):
            return False
    return True

def install_data_processing_dependencies(pip_path):
    """Install data processing dependencies"""
    data_packages = [
        "numpy==2.3.1",
        "pandas==2.1.4",
        "scikit-learn==1.3.2"
    ]
    
    for package in data_packages:
        if not run_command(f"{pip_path} install {package}", f"Installing {package}"):
            return False
    return True

def install_development_dependencies(pip_path):
    """Install development dependencies (optional)"""
    dev_packages = [
        "pytest==7.4.3",
        "black==23.11.0",
        "flake8==6.1.0",
        "mypy==1.7.1"
    ]
    
    print("\n📦 Installing development dependencies (optional)...")
    for package in dev_packages:
        run_command(f"{pip_path} install {package}", f"Installing {package}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    print("🔄 Creating .env file...")
    env_content = """# ===============================================================================
# ENVIRONMENT VARIABLES FOR RAG SYSTEM
# ===============================================================================

# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=app.log

# Document Processing
MAX_FILE_SIZE=104857600  # 100MB in bytes
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store Configuration
VECTOR_DIMENSION=768
SIMILARITY_METHOD=cosine
TOP_K_RESULTS=5

# ===============================================================================
# NOTES:
# ===============================================================================
# 1. Replace 'your_gemini_api_key_here' with your actual Gemini API key
# 2. Get API key from: https://makersuite.google.com/app/apikey
# 3. Adjust chunk size and overlap based on your document types
# 4. Set DEBUG=true for development, false for production
# ===============================================================================
"""
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        print("⚠️  IMPORTANT: Update GEMINI_API_KEY in .env file with your actual API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def verify_installation(python_path):
    """Verify that all dependencies are installed correctly"""
    print("\n🔍 Verifying installation...")
    
    test_imports = [
        "fastapi",
        "uvicorn",
        "google.generativeai",
        "numpy",
        "PyPDF2",
        "python-docx",
        "openpyxl",
        "PIL"
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError as e:
            print(f"❌ {module} - FAILED: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} modules failed to import:")
        for module in failed_imports:
            print(f"   - {module}")
        return False
    
    print("\n✅ All dependencies verified successfully!")
    return True

def install_nltk_data(python_path):
    """Install NLTK data for text processing"""
    print("\n📚 Installing NLTK data...")
    
    nltk_script = f"""
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK data installed successfully")
except Exception as e:
    print(f"Failed to install NLTK data: {{e}}")
"""
    
    try:
        result = subprocess.run([python_path, "-c", nltk_script], 
                              capture_output=True, text=True, check=True)
        print("✅ NLTK data installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install NLTK data: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 RAG SYSTEM DEPENDENCY INSTALLER")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Get paths for activated environment
    pip_path, python_path = activate_virtual_environment()
    
    print(f"\n📦 Using pip: {pip_path}")
    print(f"🐍 Using python: {python_path}")
    
    # Upgrade pip
    if not upgrade_pip(pip_path):
        print("❌ Failed to upgrade pip")
        sys.exit(1)
    
    # Install dependencies in order
    print("\n📦 Installing core dependencies...")
    if not install_core_dependencies(pip_path):
        print("❌ Failed to install core dependencies")
        sys.exit(1)
    
    print("\n🤖 Installing Google Gemini dependencies...")
    if not install_google_gemini_dependencies(pip_path):
        print("❌ Failed to install Gemini dependencies")
        sys.exit(1)
    
    print("\n📄 Installing document processing dependencies...")
    if not install_document_processing_dependencies(pip_path):
        print("❌ Failed to install document processing dependencies")
        sys.exit(1)
    
    print("\n🔢 Installing data processing dependencies...")
    if not install_data_processing_dependencies(pip_path):
        print("❌ Failed to install data processing dependencies")
        sys.exit(1)
    
    # Install development dependencies (optional)
    install_development_dependencies(pip_path)
    
    # Install NLTK data
    install_nltk_data(python_path)
    
    # Create .env file
    create_env_file()
    
    # Verify installation
    if not verify_installation(python_path):
        print("❌ Installation verification failed")
        sys.exit(1)
    
    print("\n🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\n📋 NEXT STEPS:")
    print("1. Update GEMINI_API_KEY in .env file with your actual API key")
    print("2. Activate virtual environment:")
    
    if platform.system().lower() == "windows":
        print("   env\\Scripts\\activate")
    else:
        print("   source env/bin/activate")
    
    print("3. Run the backend:")
    print("   python main.py")
    print("\n4. Open browser to: http://localhost:8000")
    print("\n📚 For more information, see QUICK_START.md")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("- If you get import errors, make sure virtual environment is activated")
    print("- If Gemini API fails, check your API key in .env file")
    print("- For other issues, check the error messages above")

if __name__ == "__main__":
    main()
