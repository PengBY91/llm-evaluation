#!/usr/bin/env python3
"""
Migration script to update existing model configurations from model_type to backend_type.

This script migrates:
- model_type="openai-chat-completions" -> backend_type="openai-api"
- model_type="openai-completions" -> backend_type="openai-api"  
- model_type="hf" -> backend_type="huggingface"
"""
import json
from pathlib import Path


# Model type mapping
TYPE_MAPPING = {
    "openai-chat-completions": "openai-api",
    "openai-completions": "openai-api",
    "hf": "huggingface",
}


def migrate_model_file(model_file: Path) -> bool:
    """
    Migrate a single model JSON file.
    
    Returns:
        True if migration was needed and successful, False if no migration needed
    """
    try:
        with open(model_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if already migrated
        if "backend_type" in data:
            print(f"  ✓ {model_file.name} already migrated")
            return False
        
        # Check if has old model_type field
        if "model_type" not in data:
            print(f"  ⚠ {model_file.name} has no model_type or backend_type field")
            return False
        
        old_type = data["model_type"]
        new_type = TYPE_MAPPING.get(old_type, old_type)
        
        # Migrate
        data["backend_type"] = new_type
        del data["model_type"]
        
        # Write back
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ {model_file.name}: {old_type} -> {new_type}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error migrating {model_file.name}: {e}")
        return False


def main():
    """Main migration function."""
    print("=" * 60)
    print("Model Configuration Migration Script")
    print("From: model_type -> To: backend_type")
    print("=" * 60)
    
    # Find models directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "apps" / "models"
    
    if not models_dir.exists():
        print(f"\n✗ Models directory not found: {models_dir}")
        print("  Please run this script from the project root directory")
        return 1
    
    # Find all JSON files
    model_files = list(models_dir.glob("*.json"))
    
    if not model_files:
        print(f"\n✓ No model files found in {models_dir}")
        return 0
    
    print(f"\nFound {len(model_files)} model file(s) in {models_dir}")
    print("\nMigrating...\n")
    
    migrated_count = 0
    for model_file in model_files:
        if migrate_model_file(model_file):
            migrated_count += 1
    
    print("\n" + "=" * 60)
    print(f"Migration complete: {migrated_count}/{len(model_files)} files migrated")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
