"""
Setup script for Gemini RAG Backend
This script helps you configure and verify the setup
"""

import os
import sys


def check_env_file():
    """Check if .env file exists and has GEMINI_API_KEY"""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("\nCreating .env file template...")
        
        try:
            with open('.env', 'w') as f:
                f.write("# Gemini API Configuration\n")
                f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
                f.write("\n# Get your API key from: https://ai.google.dev/\n")
            print("✓ Created .env file")
            print("\n⚠️  Please edit .env and add your actual GEMINI_API_KEY")
            return False
        except Exception as e:
            print(f"✗ Failed to create .env file: {e}")
            return False
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("❌ GEMINI_API_KEY not set in .env file")
        print("\n⚠️  Please edit .env and add your actual GEMINI_API_KEY")
        print("   Get your API key from: https://ai.google.dev/")
        return False
    
    print("✓ .env file configured with API key")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = {
        'google.genai': 'google-genai',
        'dotenv': 'python-dotenv',
        'numpy': 'numpy',
        'asyncio': 'built-in'
    }
    
    missing = []
    
    for module, package in required_packages.items():
        if module == 'asyncio':
            continue  # built-in
        
        module_name = module.split('.')[0]
        try:
            __import__(module_name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def check_metadata():
    """Check if chunk metadata file exists"""
    print("\nChecking chunk metadata...")
    
    if not os.path.exists('cpc_metadata.json'):
        print("❌ cpc_metadata.json not found")
        print("\n⚠️  Please run embedding_comparison.py first to generate chunks:")
        print("   python embedding_comparison.py")
        return False
    
    import json
    try:
        with open('cpc_metadata.json', 'r') as f:
            chunks = json.load(f)
        print(f"✓ Found {len(chunks)} chunks in metadata file")
        return True
    except Exception as e:
        print(f"❌ Error reading cpc_metadata.json: {e}")
        return False


def check_prompts():
    """Check if prompts.py exists"""
    print("\nChecking prompts file...")
    
    if not os.path.exists('prompts.py'):
        print("⚠️  prompts.py not found - will use fallback prompts")
        return True  # Not critical, fallback exists
    
    try:
        from prompts import response_prompt
        print("✓ prompts.py found and loaded")
        return True
    except Exception as e:
        print(f"⚠️  Error loading prompts.py: {e}")
        print("   Will use fallback prompts")
        return True  # Not critical


def main():
    print("="*80)
    print("GEMINI RAG BACKEND - SETUP VERIFICATION")
    print("="*80)
    
    checks = []
    
    # Run all checks
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("Environment", check_env_file()))
    checks.append(("Metadata", check_metadata()))
    checks.append(("Prompts", check_prompts()))
    
    # Summary
    print("\n" + "="*80)
    print("SETUP SUMMARY")
    print("="*80)
    
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to run:")
        print("   python test_gemini_rag.py")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nQuick setup steps:")
        print("  1. pip install -r requirements.txt")
        print("  2. Edit .env and add GEMINI_API_KEY")
        print("  3. python embedding_comparison.py (if needed)")
        print("  4. python test_gemini_rag.py")
    
    print("="*80)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
