import json
path = r'c:\Users\amart\Desktop\LANGUAGE_TRANSLATER\kaggle\notebook_2_train_nmt.ipynb'
nb = json.load(open(path))
print("nbformat:", nb["nbformat"], nb["nbformat_minor"])
print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    ct = c["cell_type"]
    print("  Cell", i+1, ":", ct)
print("Valid JSON: OK")
