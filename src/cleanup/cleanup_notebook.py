import sys
import os
import json

def resolve_git_conflicts(raw_text):
    """
    Scans raw text for Git merge conflict markers.
    Resolves the conflicts by keeping the local (HEAD) block
    and discarding the conflicting remote block to restore JSON structure.
    """
    lines = raw_text.splitlines()
    cleaned_lines = []
    in_conflict = False
    keep_lines = True  # Keep the local (top) block of the conflict
    conflict_count = 0

    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith("<<<<<<<"):
            in_conflict = True
            keep_lines = True  # We enter the conflict; keep the HEAD changes
            conflict_count += 1
            continue
        elif stripped.startswith("======="):
            keep_lines = False  # Discard the incoming remote changes
            continue
        elif stripped.startswith(">>>>>>>"):
            in_conflict = False
            keep_lines = True  # Resume normal reading
            continue

        if not in_conflict:
            cleaned_lines.append(line)
        else:
            if keep_lines:
                cleaned_lines.append(line)

    if conflict_count > 0:
        print(f"-> Detected and resolved {conflict_count} Git conflict block(s).")
    return "\n".join(cleaned_lines)

def strip_outputs_and_metadata(notebook_dict):
    """
    Strips bloated execution outputs, execution counts, and non-essential
    metadata to minimize file footprint and prevent memory overflows in Kaggle.
    """
    if "cells" not in notebook_dict:
        return notebook_dict

    code_cells_cleaned = 0
    for cell in notebook_dict["cells"]:
        if cell.get("cell_type") == "code":
            # Clear heavy outputs (images, base64 data, large printouts)
            cell["outputs"] = []
            cell["execution_count"] = None
            code_cells_cleaned += 1
            
        # Clear local cell metadata to prevent conflict leaks
        if "metadata" in cell:
            cell["metadata"] = {}

    # Strip global notebook metadata down to the essentials (kernel & language)
    if "metadata" in notebook_dict:
        essential_keys = ["kernelspec", "language_info"]
        notebook_dict["metadata"] = {
            k: notebook_dict["metadata"][k] 
            for k in essential_keys 
            if k in notebook_dict["metadata"]
        }

    print(f"-> Cleaned outputs and metadata for {code_cells_cleaned} code cells.")
    return notebook_dict

def repair_notebook(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        sys.exit(1)

    print(f"Processing '{file_path}'...")

    # 1. Read as raw text to resolve VCS markers
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    resolved_text = resolve_git_conflicts(raw_content)

    # 2. Parse as JSON
    try:
        notebook_json = json.loads(resolved_text)
    except json.JSONDecodeError as e:
        print(f"\n[CRITICAL ERROR] JSON parsing failed even after VCS cleanup!")
        print(f"Error details: {e}")
        print("This usually means a manual edit severed a JSON bracket or comma outside the conflict zones.")
        sys.exit(1)

    # 3. Strip bloat and clean metadata
    cleaned_json = strip_outputs_and_metadata(notebook_json)

    # 4. Save back to the same file (In-place rewrite)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_json, f, indent=1, ensure_ascii=False)

    print(f"Success! '{file_path}' has been fully repaired and compressed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python repair_notebook.py <path_to_notebook.ipynb>")
        sys.exit(1)
        
    repair_notebook(sys.argv[1])