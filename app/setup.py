"""Quick start setup script."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup routine."""
    app_dir = Path(__file__).parent
    os.chdir(app_dir)
    
    print("\n" + "="*50)
    print("RAG AI Chat - Quick Setup")
    print("="*50)
    
    # Step 1: Virtual environment
    print("\n📦 Setting up Python environment...")
    if not (app_dir / ".venv").exists():
        run_command("python3 -m venv .venv", "Create virtual environment")
    
    # Step 2: Install dependencies
    activate_cmd = "source .venv/bin/activate"
    run_command(
        f"{activate_cmd} && pip install --upgrade pip",
        "Upgrade pip"
    )
    run_command(
        f"{activate_cmd} && pip install -r requirements.txt",
        "Install dependencies"
    )
    
    # Step 3: Check .env
    print("\n⚙️  Checking configuration...")
    if not (app_dir / ".env").exists():
        print("⚠️  .env file not found. Creating from template...")
        # Copy or create .env
        env_content = """# Ollama Configuration
EMBED_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text-v2-moe
GEN_BASE_URL=http://localhost:11434/v1
GEN_MODEL=gemma4:12b
GEN_API_KEY=ollama

# PostgreSQL Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_ai

# RAG Parameters
TOP_K=5
CHUNK_SIZE=700
CHUNK_OVERLAP=120
"""
        (app_dir / ".env").write_text(env_content)
        print("✅ .env created. Please update DATABASE_URL with your PostgreSQL password!")
    
    # Step 4: Instructions
    print("\n" + "="*50)
    print("✅ Setup Complete!")
    print("="*50)
    
    print("""
📋 Next Steps:

1️⃣  PostgreSQL Setup (if needed):
   sudo apt-get install postgresql postgresql-contrib
   sudo systemctl start postgresql
   
   Create database:
   sudo -u postgres psql
   CREATE DATABASE rag_ai;
   \\c rag_ai
   CREATE EXTENSION vector;
   \\q

2️⃣  Ollama Setup:
   - Download from https://ollama.com
   - Run: ollama serve
   - Pull models:
     ollama pull nomic-embed-text-v2-moe
     ollama pull gemma4:12b

3️⃣  Update .env:
   Edit .env with your PostgreSQL password

4️⃣  Run Application (3 terminals):
   Terminal 1: ollama serve
   Terminal 2: source .venv/bin/activate && ./run_backend.sh
   Terminal 3: source .venv/bin/activate && ./run_frontend.sh

5️⃣  Open browser:
   http://localhost:8501 (Streamlit frontend)
   http://localhost:8000/docs (API documentation)

📚 For detailed setup, see SETUP.md
""")


if __name__ == "__main__":
    main()
