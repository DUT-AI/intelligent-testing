import json
nb = json.load(open("feature_extraction.ipynb", encoding="utf-8"))
code_cells = [ "".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code" ]
# exec setup + B1..B6 (indices 0..6), skip LLM cells (7+)
ns = {}
for i, src in enumerate(code_cells[:7]):
    print(f"\n########## CODE CELL {i} ##########")
    exec(compile(src, f"cell{i}", "exec"), ns)
