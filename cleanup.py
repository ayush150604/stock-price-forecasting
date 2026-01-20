"""
Cleanup script - removes old data files
"""
import os
import shutil

def cleanup():
    print("="*60)
    print("CLEANUP SCRIPT - DELETING ALL OLD DATA")
    print("="*60)
    print()
    
    folders_to_clean = [
        'data/raw',
        'data/processed',
        'results'
    ]
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            print(f"Cleaning {folder}...")
            
            # Count files
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            print(f"  Found {len(files)} files")
            
            # Remove all files in the folder
            for file in files:
                file_path = os.path.join(folder, file)
                try:
                    os.unlink(file_path)
                    print(f"  ✓ Deleted: {file}")
                except Exception as e:
                    print(f"  ✗ Error deleting {file}: {e}")
        else:
            print(f"Creating {folder}...")
            os.makedirs(folder, exist_ok=True)
    
    print()
    print("="*60)
    print("✅ Cleanup complete! All old files deleted.")
    print("="*60)
    print()
    print("Verifying folders are empty...")
    for folder in folders_to_clean:
        files = os.listdir(folder) if os.path.exists(folder) else []
        print(f"  {folder}: {len(files)} files")
    
    print()
    print("Now run: python run_single_stock.py")

if __name__ == "__main__":
    cleanup()