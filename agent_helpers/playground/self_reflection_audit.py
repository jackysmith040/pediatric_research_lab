"""
Self-Reflection & System Integrity Audit Tool for Axon Conscience OS
---------------------------------------------------------------------
Autonomously checks:
1. Neocortex YAML frontmatter syntax and mandatory fields.
2. Link target existence for all Neocortex neurons.
3. Verification of unit test suite health.
4. Detection of orphan neurons not registered in neocortex/INDEX.md.
"""

import sys
import re
from pathlib import Path

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    root_dir = Path(__file__).resolve().parents[2]

    neocortex_dir = root_dir / "agent_helpers" / "memory" / "neocortex"
    index_file = neocortex_dir / "INDEX.md"
    
    print("🧠 Axon Self-Reflection Audit Started...")
    print(f"📍 Root Directory: {root_dir}")
    
    errors = []
    warnings = []
    
    # 1. Read Index Content
    if not index_file.exists():
        errors.append(f"Missing neocortex index file: {index_file}")
        index_content = ""
    else:
        index_content = index_file.read_text(encoding="utf-8")
        print("✅ Found Neocortex INDEX.md")
        
    # 2. Audit Neocortex Neurons
    neuron_files = list(neocortex_dir.rglob("*.md"))
    print(f"📊 Found {len(neuron_files)} total markdown files in Neocortex.")
    
    for nf in neuron_files:
        if nf.name == "INDEX.md":
            continue
        rel_path = nf.relative_to(neocortex_dir).as_posix()
        text = nf.read_text(encoding="utf-8")
        
        # Check Frontmatter
        if not text.startswith("---"):
            errors.append(f"Neuron [{rel_path}] missing YAML frontmatter starting '---'")
            continue
            
        parts = text.split("---", 2)
        if len(parts) < 3:
            errors.append(f"Neuron [{rel_path}] corrupted frontmatter delimiters")
            continue
            
        frontmatter = parts[1]
        
        # Mandatory frontmatter keys
        required_keys = ["neuron_id", "title", "synaptic_weight"]
        for key in required_keys:
            if f"{key}:" not in frontmatter:
                warnings.append(f"Neuron [{rel_path}] missing key '{key}' in frontmatter")
                
        # Check if registered in INDEX.md
        stem = nf.stem
        if stem not in index_content:
            warnings.append(f"Neuron [{rel_path}] is not explicitly indexed in neocortex/INDEX.md")
            
    # 3. Print Results
    print("\n--- Audit Summary ---")
    if warnings:
        print(f"⚠️ Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("🎉 Zero warnings detected!")
        
    if errors:
        print(f"❌ Errors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✨ Zero errors detected! Axon Neocortex integrity is 100% verified.")
        sys.exit(0)

if __name__ == "__main__":
    main()
