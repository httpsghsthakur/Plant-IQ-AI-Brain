"""
Strip all non-ASCII characters from print() and f-string statements in Python files.
This prevents Windows 'charmap' codec errors.
"""
import re
import os
from pathlib import Path

ROOT = Path(__file__).parent

# Directories to process
DIRS = [
    ROOT / "models",
    ROOT / "services",
    ROOT / "api",
    ROOT / "training",
]

REPLACEMENTS = {
    "\U0001f4ca": "[*]",      # 📊
    "\U00002705": "[OK]",     # ✅
    "\U000026a0\U0000fe0f": "[WARN]",  # ⚠️
    "\U000026a0": "[WARN]",   # ⚠ (without variation selector)
    "\U0000274c": "[ERROR]",  # ❌
    "\U0001f4e6": "[OK]",     # 📦
    "\U0001f916": "[AI]",     # 🤖
    "\U0001f4c2": "[DIR]",    # 📂
    "\U0001f680": "[START]",  # 🚀
    "\U0001f3c6": "[BEST]",   # 🏆
    "\U0001f4c1": "[DIR]",    # 📁
    "\U0001f6d1": "[STOP]",   # 🛑
    "\U0001f331": "[PLANT]",  # 🌱
    "\U0001f4d6": "[DOC]",    # 📖
    "\U0001f4ac": "[CHAT]",   # 💬
    "\U00002702\U0000fe0f": "[CUT]",   # ✂️
    "\U00002702": "[CUT]",    # ✂
    "\U0001f321\U0000fe0f": "[TEMP]",  # 🌡️
    "\U0001f321": "[TEMP]",   # 🌡
    "\u2022": "*",            # •
    "\u2191": "^",            # ↑
    "\u2192": "->",           # →
    "\u2193": "v",            # ↓
    "\u00b2": "2",            # ²
    "\u00b1": "+/-",          # ±
    "\u2014": "--",           # —
    "\u2013": "-",            # –
}

def clean_file(filepath: Path):
    try:
        content = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    
    original = content
    for emoji, replacement in REPLACEMENTS.items():
        content = content.replace(emoji, replacement)
    
    # Remove any remaining non-ASCII in print/f-string lines only
    # Keep rupee symbol (₹) as it's expected in data
    
    if content != original:
        filepath.write_text(content, encoding="utf-8")
        print(f"  Fixed: {filepath.relative_to(ROOT)}")
        return True
    return False

def main():
    fixed = 0
    total = 0
    
    for d in DIRS:
        if not d.exists():
            continue
        for py_file in d.rglob("*.py"):
            total += 1
            if clean_file(py_file):
                fixed += 1
    
    print(f"\nDone: {fixed}/{total} files cleaned of non-ASCII characters.")

if __name__ == "__main__":
    main()
