import ast
from pathlib import Path

def audit_ui_element_access(file_path: Path):
    print("=" * 60)
    print(f"AUDITING UI ELEMENT USAGE IN: {file_path.name}")
    print("=" * 60)
    code = file_path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    
    issues = []
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
                cell_name = f"cell-{cell_idx} (line {node.lineno})"
                
                # Find all variables assigned to mo.ui.* in this cell
                created_uis = set()
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        val = stmt.value
                        if isinstance(val, ast.Call):
                            func = val.func
                            if isinstance(func, ast.Attribute) and isinstance(func.value, (ast.Attribute, ast.Name)):
                                # mo.ui.slider or ui.slider
                                if getattr(func.value, "attr", "") == "ui" or getattr(func.value, "id", "") == "ui":
                                    for t in stmt.targets:
                                        if isinstance(t, ast.Name):
                                            created_uis.add(t.id)
                
                # Now check if any of these created_uis have .value accessed in this same cell
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and sub.attr == "value":
                        if isinstance(sub.value, ast.Name) and sub.value.id in created_uis:
                            issues.append((cell_name, sub.value.id, sub.lineno))
                            
    if issues:
        print(f"FOUND {len(issues)} SAME-CELL UI .value ACCESS VIOLATIONS:")
        for cname, var, lineno in issues:
            print(f"  [X] {cname}: UI element '{var}' accessed with .value at line {lineno} in the cell that created it!")
    else:
        print("NO SAME-CELL UI .value ACCESS VIOLATIONS FOUND!")
    print()

if __name__ == "__main__":
    audit_ui_element_access(Path("distillation_notebook/pediatric_vision_lab.py"))
    audit_ui_element_access(Path("distillation_notebook/dinov3_yolo26s_distillation_train.py"))
    audit_ui_element_access(Path("distillation_notebook/traditional_yolo26s_finetune_train.py"))
