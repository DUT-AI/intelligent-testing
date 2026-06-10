import json
nb = json.load(open("feature_extraction.ipynb", encoding="utf-8"))
code_cells = [ "".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code" ]
ns = {}
for i in (0, 1, 7):   # setup, split, LLM client + demo call
    print(f"\n##### CODE CELL {i} #####")
    exec(compile(code_cells[i], f"cell{i}", "exec"), ns)
