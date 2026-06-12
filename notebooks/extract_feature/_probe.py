import json, requests
nb = json.load(open("feature_extraction.ipynb", encoding="utf-8"))
code_cells = [ "".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code" ]
ns={}
for i in (0,1): exec(code_cells[i], ns)
# build prompt for Q10
exec(code_cells[7].split("# Test thử")[0], ns)  # define client + build_prompt, skip demo
df=ns["df"]; build_prompt=ns["build_prompt"]
row=df[df.question_id==10].iloc[0]
prompt=build_prompt(row)
for mt in (1024, 4096):
    r=requests.post(f'{ns["LLM_BASE"]}/chat/completions', headers=ns["LLM_HEADERS"],
        json={"model":ns["LLM_MODEL"],"messages":[
          {"role":"system","content":"Bạn là chuyên gia phân tích độ khó câu hỏi lập trình C++."},
          {"role":"user","content":prompt}],"temperature":0.0,"max_tokens":mt}, timeout=180)
    j=r.json(); m=j["choices"][0]
    print(f"\n===== max_tokens={mt} finish={m['finish_reason']} =====")
    print("content len:", len(m["message"].get("content") or ""))
    print("reasoning len:", len(m["message"].get("reasoning_content") or ""))
    print("CONTENT:", repr((m["message"].get("content") or "")[:800]))
