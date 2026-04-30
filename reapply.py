import json

# 04
with open('notebooks/04_cross_model_comparison.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "figsize=(18, 10)" in line and "sharex" not in line:
                source[i] = line.replace("figsize=(18, 10)", "figsize=(18, 10), sharex=True")
                break
with open('notebooks/04_cross_model_comparison.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

# 02
with open('notebooks/02_calibration_curves.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        inserted = False
        for i, line in enumerate(source):
            if "ax.legend(" in line and not inserted:
                # check if xlim is already there
                if any("set_xlim" in l for l in source):
                    break
                source.insert(i, "    ax.set_xlim(0.25, 1.0)\n")
                inserted = True
                break
with open('notebooks/02_calibration_curves.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

# gitignore
with open('.gitignore', 'a') as f:
    f.write(".vscode/\n")
