import ast
from pathlib import Path
from collections import defaultdict

def audit_marimo_deep(file_path: Path):
    print("=" * 70)
    print(f"DEEP AUDIT FOR: {file_path.name}")
    print("=" * 70)
    
    code = file_path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    
    defined_in_cells = defaultdict(list)
    cell_idx = 0
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            is_cell = any(
                (isinstance(dec, ast.Attribute) and dec.attr == "cell") or 
                (isinstance(dec, ast.Name) and dec.id == "cell")
                for dec in node.decorator_list
            )
            
            if is_cell:
                cell_idx += 1
                cell_name = f"cell-{cell_idx} (def {node.name} at line {node.lineno})"
                cell_vars = set()
                
                for sub in ast.walk(node):
                    # Targets in assignment
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            for name in ast.walk(target):
                                if isinstance(name, ast.Name):
                                    cell_vars.add(name.id)
                    elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                        cell_vars.add(sub.target.id)
                    elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name):
                        cell_vars.add(sub.target.id)
                    elif isinstance(sub, ast.For):
                        for name in ast.walk(sub.target):
                            if isinstance(name, ast.Name):
                                cell_vars.add(name.id)
                    elif isinstance(sub, ast.With):
                        for item in sub.items:
                            if item.optional_vars:
                                for name in ast.walk(item.optional_vars):
                                    if isinstance(name, ast.Name):
                                        cell_vars.add(name.id)
                    elif isinstance(sub, ast.FunctionDef) and sub != node:
                        cell_vars.add(sub.name)
                    elif isinstance(sub, ast.ClassDef):
                        cell_vars.add(sub.name)
                
                # Filter out variables starting with _ or builtins/comprehension vars if any
                for var in cell_vars:
                    if not var.startswith("_"):
                        defined_in_cells[var].append(cell_name)
                        
    duplicates = {var: cells for var, cells in defined_in_cells.items() if len(cells) > 1}
    
    if duplicates:
        print(f"FOUND {len(duplicates)} REDEFINED VARIABLES ACROSS CELLS:")
        for var, cells in sorted(duplicates.items()):
            print(f"\n  [X] Variable '{var}' defined in {len(cells)} cells:")
            for c in cells:
                print(f"       -> {c}")
    else:
        print("NO REDEFINED VARIABLES FOUND!")
    print()

if __name__ == "__main__":
    audit_marimo_deep(Path("distillation_notebook/pediatric_vision_lab.py"))
    audit_marimo_deep(Path("distillation_notebook/dinov3_yolo26s_distillation_train.py"))
    audit_marimo_deep(Path("distillation_notebook/traditional_yolo26s_finetune_train.py"))
