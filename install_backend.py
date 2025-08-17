#!/usr/bin/env python3
"""
===============================================================================
AUTOMATED BACKEND INSTALLATION SCRIPT
===============================================================================

PURPOSE:
This script automatically sets up the complete backend environment for the RAG system
including virtual environment, dependencies, and verification.

WHAT IT DOES:
1. Creates virtual environment
2. Installs all required packages
3. Downloads AI models
4. Verifies installation
5. Sets up environment variables

USAGE:
python install_backend.py

REQUIREMENTS:
- Python 3.8+ installed
- pip available
- Internet connection for downloads
===============================================================================
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# ===============================================================================
# CONFIGURATION
# ===============================================================================
VENV_NAME = "env"
REQUIREMENTS_FILE = "requirements.txt"
PYTHON_VERSION_MIN = (3, 8)

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_step(step, description):
    """Print a step with description"""
    print(f"\n🔧 STEP {step}: {description}")
    print("-" * 60)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️ {message}")

def run_command(command, check=True, capture_output=False):
    """Run a shell command and return result"""
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, check=check, 
                                 capture_output=True, text=True)
            return result
        else:
            result = subprocess.run(command, shell=True, check=check)
            return result
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Command failed: {command}")
            print_error(f"Error: {e}")
            sys.exit(1)
        return e

def check_python_version():
    """Check if Python version meets requirements"""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < PYTHON_VERSION_MIN:
        print_error(f"Python {PYTHON_VERSION_MIN[0]}.{PYTHON_VERSION_MIN[1]}+ required")
        sys.exit(1)
    
    print_success(f"Python version {version.major}.{version.minor} is compatible")
    return True

def check_pip():
    """Check if pip is available"""
    print_step(2, "Checking pip availability")
    
    try:
        result = run_command("pip --version", capture_output=True)
        print_success(f"pip found: {result.stdout.strip()}")
        return True
    except:
        print_error("pip not found. Please install pip first.")
        sys.exit(1)

def create_virtual_environment():
    """Create virtual environment"""
    print_step(3, "Creating Virtual Environment")
    
    # Remove existing environment if it exists
    if os.path.exists(VENV_NAME):
        print_warning(f"Virtual environment '{VENV_NAME}' already exists")
        response = input("Do you want to recreate it? (y/N): ").strip().lower()
        if response == 'y':
            print("Removing existing virtual environment...")
            shutil.rmtree(VENV_NAME)
        else:
            print_success("Using existing virtual environment")
            return True
    
    print("Creating new virtual environment...")
    run_command(f"python -m venv {VENV_NAME}")
    print_success(f"Virtual environment '{VENV_NAME}' created successfully")
    return True

def get_activate_command():
    """Get the appropriate activation command for the current OS"""
    system = platform.system().lower()
    
    if system == "windows":
        if "powershell" in os.environ.get("SHELL", "").lower():
            return f"{VENV_NAME}\\Scripts\\Activate.ps1"
        else:
            return f"{VENV_NAME}\\Scripts\\activate.bat"
    else:
        return f"source {VENV_NAME}/bin/activate"

def upgrade_pip():
    """Upgrade pip to latest version"""
    print_step(4, "Upgrading pip")
    
    activate_cmd = get_activate_command()
    upgrade_cmd = f"{activate_cmd} && python -m pip install --upgrade pip"
    
    print("Upgrading pip...")
    run_command(upgrade_cmd)
    print_success("pip upgraded successfully")

def install_requirements():
    """Install all required packages"""
    print_step(5, "Installing Dependencies")
    
    if not os.path.exists(REQUIREMENTS_FILE):
        print_error(f"Requirements file '{REQUIREMENTS_FILE}' not found")
        sys.exit(1)
    
    activate_cmd = get_activate_command()
    install_cmd = f"{activate_cmd} && pip install -r {REQUIREMENTS_FILE}"
    
    print("Installing packages (this may take several minutes)...")
    print("Downloading PyTorch and AI models (~2-3 GB)...")
    
    try:
        run_command(install_cmd)
        print_success("All packages installed successfully")
    except:
        print_error("Package installation failed")
        print_warning("Try running manually: pip install -r requirements.txt")
        sys.exit(1)

def verify_installation():
    """Verify that all critical packages are installed correctly"""
    print_step(6, "Verifying Installation")
    
    activate_cmd = get_activate_command()
    
    # Test imports
    test_script = f"""
{activate_cmd}
python -c "
try:
    import fastapi
    print('✅ FastAPI:', fastapi.__version__)
    
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('   CUDA available:', torch.cuda.is_available())
    
    import sentence_transformers
    print('✅ Sentence Transformers:', sentence_transformers.__version__)
    
    import numpy
    print('✅ NumPy:', numpy.__version__)
    
    print('\\n🎉 All critical packages installed successfully!')
    
except ImportError as e:
    print('❌ Import failed:', e)
    exit(1)
"
"""
    
    print("Testing package imports...")
    run_command(test_script)
    print_success("Package verification completed")

def download_ai_models():
    """Download and cache AI models"""
    print_step(7, "Downloading AI Models")
    
    activate_cmd = get_activate_command()
    
    # Download sentence transformer models
    download_script = f"""
{activate_cmd}
python -c "
from sentence_transformers import SentenceTransformer
print('📥 Downloading primary model: all-MiniLM-L6-v2')
model1 = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Primary model downloaded')

print('📥 Downloading fallback model: paraphrase-MiniLM-L3-v2')
model2 = SentenceTransformer('paraphrase-MiniLM-L3-v2')
print('✅ Fallback model downloaded')

print('🎯 All AI models are ready!')
"
"""
    
    print("Downloading AI models (this may take a few minutes)...")
    run_command(download_script)
    print_success("AI models downloaded and cached")

def create_environment_file():
    """Create .env file with basic configuration"""
    print_step(8, "Creating Environment Configuration")
    
    env_content = """# ===============================================================================
# ENVIRONMENT VARIABLES FOR RAG SYSTEM
# ===============================================================================

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEVICE=auto

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security (change these in production!)
SECRET_KEY=your-secret-key-here-change-in-production
API_KEY=your-api-key-here

# Optional: Vector Database (Pinecone)
# PINECONE_API_KEY=your-pinecone-api-key
# PINECONE_ENVIRONMENT=your-pinecone-environment
# PINECONE_INDEX_NAME=your-index-name

# Optional: OpenAI (if using GPT models)
# OPENAI_API_KEY=your-openai-api-key
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print_success(".env file created with basic configuration")
    print_warning("Remember to update SECRET_KEY and API_KEY for production!")

def create_startup_script():
    """Create startup script for easy server launch"""
    print_step(9, "Creating Startup Scripts")
    
    # Windows batch file
    if platform.system().lower() == "windows":
        batch_content = f"""@echo off
echo Starting RAG System Backend...
echo.
call {VENV_NAME}\\Scripts\\activate.bat
echo Virtual environment activated
echo.
echo Starting FastAPI server...
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
"""
        
        with open("start_backend.bat", "w") as f:
            f.write(batch_content)
        print_success("start_backend.bat created")
    
    # Unix shell script
    shell_content = f"""#!/bin/bash
echo "Starting RAG System Backend..."
echo ""
source {VENV_NAME}/bin/activate
echo "Virtual environment activated"
echo ""
echo "Starting FastAPI server..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
    
    with open("start_backend.sh", "w") as f:
        f.write(shell_content)
    
    # Make shell script executable
    if platform.system().lower() != "windows":
        os.chmod("start_backend.sh", 0o755)
    
    print_success("start_backend.sh created")

def print_final_instructions():
    """Print final setup instructions"""
    print_header("INSTALLATION COMPLETE!")
    
    print("🎉 Your RAG system backend is now ready!")
    print("\n📋 NEXT STEPS:")
    print("1. Start the backend server:")
    
    if platform.system().lower() == "windows":
        print("   Double-click: start_backend.bat")
        print("   Or run: .\\env\\Scripts\\activate && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("   Run: ./start_backend.sh")
        print("   Or run: source env/bin/activate && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    
    print("\n2. Access your API:")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    
    print("\n3. Test the system:")
    print("   - Upload a document via the API")
    print("   - Ask questions and get AI-powered answers")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("- The system will download AI models on first use (~100MB)")
    print("- GPU acceleration will be used automatically if available")
    print("- Check .env file and update security keys for production")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("- If you get import errors, make sure virtual environment is activated")
    print("- If models fail to download, check your internet connection")
    print("- For GPU issues, ensure PyTorch CUDA is properly installed")

# ===============================================================================
# MAIN INSTALLATION PROCESS
# ===============================================================================

def main():
    """Main installation function"""
    print_header("RAG SYSTEM BACKEND INSTALLER")
    print("This script will set up your complete backend environment")
    
    try:
        # Run all installation steps
        check_python_version()
        check_pip()
        create_virtual_environment()
        upgrade_pip()
        install_requirements()
        verify_installation()
        download_ai_models()
        create_environment_file()
        create_startup_script()
        
        print_final_instructions()
        
    except KeyboardInterrupt:
        print("\n\n❌ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Installation failed: {e}")
        print("Please check the error messages above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
