from pathlib import Path
import hashlib
import json
import subprocess

OLD = Path("/Users/cesar/Eldorado/vrptw")
NEW = Path("/Users/cesar/Eldorado/walsh_definitivo")
OUT = NEW / "reference" / "legacy"

OUT.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git(cmd, root=OLD):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root)] + cmd,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""

files = []

def add(path, category):
    if path.is_file():
        files.append({
            "category": category,
            "path": str(path.relative_to(OLD)),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })

# ============================================================
# 1. Arquivos raiz importantes
# ============================================================

for name in [
    "experiment_1_full_walsh_random.py",
    "requirements.txt",
    "pyproject.toml",
    "environment.yml",
    "environment.yaml",
]:
    add(OLD / name, "root")

# ============================================================
# 2. Todo código/config relevante dentro de artigo/
# ============================================================

allowed = {
    ".py", ".ipynb", ".yaml", ".yml", ".json",
    ".toml", ".tex", ".bib", ".md", ".sh"
}

artigo = OLD / "artigo"

if artigo.exists():
    for p in artigo.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in allowed
            and "__pycache__" not in p.parts
            and ".ipynb_checkpoints" not in p.parts
        ):
            add(p, "artigo")

# ============================================================
# 3. Biblioteca central
# ============================================================

src = OLD / "src" / "walsh_pce"

if src.exists():
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in {
            ".py", ".yaml", ".yml", ".json", ".toml"
        }:
            add(p, "src")

# ============================================================
# 4. Resultados consolidados importantes
# ============================================================

results = OLD / "results"

keywords = [
    "summary",
    "consolidated",
    "manifest",
    "metadata",
    "calibration",
    "hyperparameter",
    "epsilon",
    "shot_sweep",
    "best_seed",
    "transpiled_resources",
]

exact = {
    "stage1_consolidated_runs.csv",
    "original_base_sweep_all_runs.csv",
    "original_base_sweep_summary.csv",
    "original_base_sweep_plan.csv",
    "final_benchmark_summary.csv",
    "final_benchmark_best_seeds.csv",
    "epsilon_exact_all.csv",
    "statevector_summary.csv",
    "transpiled_resources_selected_seeds.csv",
    "hw_n15_problem.json",
    "raw_32768_10seeds_summary.csv",
}

if results.exists():
    for p in results.rglob("*"):
        if not p.is_file():
            continue

        name = p.name.lower()

        if (
            p.name in exact
            or any(k in name for k in keywords)
        ) and p.suffix.lower() in {
            ".csv", ".json", ".yaml", ".yml", ".txt"
        }:
            add(p, "results")

# ============================================================
# 5. Dados usados diretamente nas figuras/tabelas
# ============================================================

analysis = OLD / "artigo" / "analysis"

if analysis.exists():
    for p in analysis.rglob("*"):
        if p.is_file() and p.suffix.lower() in {
            ".py", ".csv", ".json", ".ipynb"
        }:
            add(p, "analysis")

# ============================================================
# 6. Gset
# ============================================================

for name in ["G14", "G23", "G60", "G64"]:
    for base in [OLD / "data", OLD / "artigo", OLD / "results"]:
        if not base.exists():
            continue

        for p in base.rglob(name):
            if p.is_file():
                add(p, "gset")

# ============================================================
# 7. MQLib
# ============================================================

mqlib = OLD / "MQLib" / "MQLib"

mqlib_info = {
    "exists": mqlib.exists(),
    "commit": "",
    "status": "",
    "binary": None,
}

if mqlib.exists():
    mqlib_info["commit"] = git(["rev-parse", "HEAD"], mqlib)
    mqlib_info["status"] = git(["status", "--short"], mqlib)

    binary = mqlib / "bin" / "MQLib"

    if binary.exists():
        mqlib_info["binary"] = {
            "path": str(binary.relative_to(OLD)),
            "size": binary.stat().st_size,
            "sha256": sha256(binary),
        }

# ============================================================
# 8. Git antigo
# ============================================================

git_info = {
    "branch": git(["branch", "--show-current"]),
    "head": git(["rev-parse", "HEAD"]),
    "status": git(["status", "--short"]),
    "remote": git(["remote", "-v"]),
}

# ============================================================
# 9. Salvar JSON
# ============================================================

payload = {
    "old_root": str(OLD),
    "new_root": str(NEW),
    "git": git_info,
    "mqlib": mqlib_info,
    "files": sorted(files, key=lambda x: (x["category"], x["path"])),
}

(OUT / "LEGACY_INVENTORY.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

# ============================================================
# 10. Salvar Markdown
# ============================================================

lines = [
    "# Legacy Walsh-PCE inventory",
    "",
    f"Old root: `{OLD}`",
    f"New root: `{NEW}`",
    "",
    "## Git",
    "",
    f"- branch: `{git_info['branch']}`",
    f"- HEAD: `{git_info['head']}`",
    "",
    "## MQLib",
    "",
    f"- exists: `{mqlib_info['exists']}`",
    f"- commit: `{mqlib_info['commit']}`",
]

if mqlib_info["binary"]:
    lines += [
        f"- binary: `{mqlib_info['binary']['path']}`",
        f"- SHA256: `{mqlib_info['binary']['sha256']}`",
    ]

categories = {}

for item in files:
    categories.setdefault(item["category"], []).append(item)

for category in sorted(categories):
    lines += ["", f"## {category}", ""]

    for item in categories[category]:
        mb = item["size"] / 1024 / 1024
        lines.append(
            f"- `{item['path']}` ({mb:.3f} MB)"
        )

(OUT / "LEGACY_INVENTORY.md").write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print()
print("Inventário criado:")
print(OUT / "LEGACY_INVENTORY.md")
print(OUT / "LEGACY_INVENTORY.json")
print()
print("Arquivos selecionados:", len(files))
