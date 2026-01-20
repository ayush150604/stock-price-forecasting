"""
Run this script to automatically set up the project structure
"""
import os

def create_directories():
    """Create all necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'utils',
        'results',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")

def create_init_files():
    """Create __init__.py files"""
    init_files = [
        'utils/__init__.py',
        'models/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('')
        print(f"✓ Created: {init_file}")

def check_files():
    """Check if all required files exist"""
    required_files = [
        'main.py',
        'requirements.txt',
        'utils/data_loader.py',
        'utils/preprocessing.py',
        'utils/evaluation.py',
        'models/arima_model.py'
    ]
    
    print("\n" + "="*50)
    print("Checking required files...")
    print("="*50)
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ Found: {file}")
        else:
            print(f"✗ Missing: {file}")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("="*50)
    print("Stock Forecasting Project Setup")
    print("="*50)
    print()
    
    # Create directories
    print("Creating directories...")
    create_directories()
    print()
    
    # Create __init__.py files
    print("Creating package files...")
    create_init_files()
    print()
    
    # Check files
    all_files_exist = check_files()
    
    print()
    print("="*50)
    if all_files_exist:
        print("✓ Setup Complete!")
        print("="*50)
        print()
        print("Next steps:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Run: python main.py")
    else:
        print("⚠ Setup Incomplete!")
        print("="*50)
        print()
        print("Please create the missing files listed above.")
    print()
    