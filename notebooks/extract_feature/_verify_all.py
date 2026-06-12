import json
nb = json.load(open("feature_extraction.ipynb", encoding="utf-8"))
code_cells = [ "".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code" ]
ns={}
for i,src in enumerate(code_cells):
    print(f"\n##### CELL {i} #####")
    exec(compile(src,f"cell{i}","exec"), ns)
