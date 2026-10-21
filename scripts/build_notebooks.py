"""Generate the pipeline notebooks in notebooks/ from the .py sources.

The .py files stay the source of truth (README commands, imports between modules, Colab runs). Each notebook copies
the code of the scripts it covers: the module docstring becomes a Markdown cell, the imports, constants and every
function get their own code cell, and a final cell sets sys.argv and calls the script's main(). Edit the .py file,
then re-run this script; never edit the notebooks by hand.

  01_data_prep.ipynb        download, index, split, waveform cache
  02_eda.ipynb              dataset analysis (Midterm part A)
  03_train_colab.ipynb      training and re-evaluation (GPU)
  04_results_midterm.ipynb  baseline table, figures, leakage check, Midterm questions E1-E4

The notebooks are written without outputs; nothing is executed here.

Usage:
    python scripts/build_notebooks.py
"""
import ast
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = '''import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/DataSEC-Sound-Classification")  # <- where the repository is on Drive
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
# os.environ["ATI_CONFIG"] = "configs/default.yaml"  # optional: another config (read when src.utils is imported)

import pandas as pd
from IPython.display import Image, display

print("repository:", ROOT)'''

COLAB_INSTALL = '''import subprocess

if IN_COLAB:  # torch, torchaudio, torchvision are preinstalled on Colab
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soxr", "thop"], check=True)'''


def show(figures: list[str] = (), tables: list[str] = ()) -> str:
    lines = ["from src.utils import RESULTS", ""]
    for t in tables:
        lines.append(f'display(pd.read_csv(RESULTS / "tables" / "{t}"))')
    for f in figures:
        lines.append(f'display(Image(filename=str(RESULTS / "figures" / "{f}")))')
    return "\n".join(lines)


def script_cells(rel: str, title: str, run: str | None) -> list:
    """Markdown + code cells for one .py file. `run` is the code of the final cell (None: no run cell)."""
    path = ROOT / rel
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    stem = path.stem
    cells = []

    body = tree.body
    doc = ast.get_docstring(tree) or ""
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    text = doc.split("\nUsage:")[0].strip()
    cells.append(new_markdown_cell(f"## {title}\n\nSource: `{rel}`\n\n{text}"))

    def segment(node: ast.AST) -> str:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        while start > 0 and lines[start - 1].lstrip().startswith("#"):  # keep comments directly above
            start -= 1
        return "\n".join(lines[start:node.end_lineno])

    pending = {"imports": [], "consts": []}

    def flush() -> None:
        for key in ("imports", "consts"):
            if pending[key]:
                cells.append(new_code_cell("\n".join(pending[key])))
                pending[key] = []

    for node in body:
        if isinstance(node, ast.If):  # if __name__ == "__main__"
            continue
        seg = segment(node)
        if re.match(r"(sys\.path\.insert|matplotlib\.use)\(", seg.strip()):
            continue  # the setup cell puts the repo on sys.path; figures are shown inline
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            pending["imports"].append(re.sub(r"\s+# noqa: E402", "", seg))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush()
            if node.name == "main":  # unique name per script, so notebook sections do not overwrite each other
                seg = seg.replace("def main(", f"def main_{stem}(", 1)
            cells.append(new_code_cell(seg))
        else:
            if pending["imports"]:
                cells.append(new_code_cell("\n".join(pending["imports"])))
                pending["imports"] = []
            pending["consts"].append(seg)
    flush()
    if run:
        cells.append(new_code_cell(run.replace("MAIN", f"main_{stem}")))
    return cells


def argv_run(rel: str, *args: str) -> str:
    cmd = " ".join(["python", rel, *args])
    argv = ", ".join(f'"{a}"' for a in [rel, *args])
    return f"# same as: {cmd}\nsys.argv = [{argv}]\nMAIN()"


def notebook(intro: str, cells: list) -> nbformat.NotebookNode:
    nb = new_notebook(cells=[new_markdown_cell(intro), new_code_cell(SETUP)] + cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


GENERATED = ("*Generated by `scripts/build_notebooks.py` from the `.py` files; edit those and rebuild, not this "
             "notebook.*")


def build() -> dict[str, nbformat.NotebookNode]:
    nbs = {}

    nbs["01_data_prep.ipynb"] = notebook(
        "# 01. Data preparation\n\n"
        "Download the pinned DataSEC release, index the clips, check the fixed split and cache the resampled "
        "waveforms used for training. Run once per machine; the cache is needed by `03_train_colab.ipynb`.\n\n"
        "The raw audio is CC-BY-NC-SA and is **not** committed (`data/` is git-ignored).\n\n" + GENERATED,
        [new_code_cell(COLAB_INSTALL)]
        + script_cells("scripts/download_data.py", "1. Download and verify (6.4 GB)",
                       argv_run("scripts/download_data.py", "--out", "data/raw", "--workers", "8"))
        + script_cells("scripts/build_index.py", "2. Index the clips", argv_run("scripts/build_index.py"))
        + script_cells("scripts/make_splits.py", "3. Fixed train / val / test split",
                       "# The committed splits/ are fixed for the Midterm and the Final: do not regenerate them.\n"
                       "REGENERATE_SPLITS = False\n"
                       "if REGENERATE_SPLITS:\n"
                       '    sys.argv = ["scripts/make_splits.py"]\n'
                       "    MAIN()\n"
                       "else:\n"
                       '    meta = pd.read_json("splits/split_meta.json", typ="series")\n'
                       '    print(meta["method"], "|", meta["groups"], "|", meta["sizes"])')
        + script_cells("scripts/cache_waveforms.py", "4. Cache 22.05 kHz waveforms",
                       "# the worker must be importable for multiprocessing on Windows / macOS (spawn)\n"
                       "from scripts.cache_waveforms import convert  # noqa: F811\n\n"
                       + argv_run("scripts/cache_waveforms.py", "--workers", "4")))

    nbs["02_eda.ipynb"] = notebook(
        "# 02. Dataset analysis (Midterm part A)\n\n"
        "Needs `data/index.csv` and the extracted audio from `01_data_prep.ipynb`. Runs on CPU. Written up in "
        "[`docs/dataset_analysis.md`](../docs/dataset_analysis.md).\n\n" + GENERATED,
        script_cells("scripts/eda.py", "1. Class balance, durations, representations, spectral similarity",
                     argv_run("scripts/eda.py"))
        + [new_code_cell(show(["eda_class_counts.png", "eda_duration_box.png", "eda_representations.png",
                               "eda_mel_filterbank.png", "eda_class_mean_spectra.png",
                               "eda_spectral_similarity.png"],
                              ["eda_duration_stats.csv", "eda_similar_pairs.csv"]))]
        + script_cells("scripts/eda_quality.py", "2. Data quality and curation artefacts",
                       "# the worker must be importable for multiprocessing on Windows / macOS (spawn)\n"
                       "from scripts.eda_quality import analyse  # noqa: F811\n\n"
                       + argv_run("scripts/eda_quality.py", "--workers", "8"))
        + [new_code_cell(show(["eda_signal_quality.png"], ["eda_overlap_pairs.csv", "eda_signal_quality.csv"]))])

    nbs["03_train_colab.ipynb"] = notebook(
        "# 03. Training (GPU)\n\n"
        "Trains `small_cnn`, `mobilenetv3s` and `crnn`. The checkpoint is selected on validation macro-F1; the test "
        "set is only scored once, after training. Each run writes `results/predictions/*_{val,test}_none_0.csv` and "
        "`results/logs/{model}_seed{seed}.json`. Midterm results: seed 42 on a Colab T4.\n\n"
        "Needs the waveform cache from `01_data_prep.ipynb` (`data/cache_22k`).\n\n" + GENERATED,
        [new_code_cell(COLAB_INSTALL),
         new_code_cell('import torch\n\nprint("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() '
                       'else "none (training on CPU is very slow)")')]
        + script_cells("src/train.py", "1. Train one model with one seed",
                       "MODELS = [\"small_cnn\", \"mobilenetv3s\", \"crnn\"]\n"
                       "SEEDS = [42]          # the Final needs [42, 43, 44]\n"
                       "RETRAIN = False       # True: overwrite runs that already have a log\n\n"
                       "# quick smoke test: python -m src.train --model crnn --seed 42 --epochs 2 --limit 256\n"
                       "for seed in SEEDS:\n"
                       "    for model_name in MODELS:\n"
                       '        if (RESULTS / "logs" / f"{model_name}_seed{seed}.json").exists() and not RETRAIN:\n'
                       '            print(f"skip {model_name} seed {seed}: log exists")\n'
                       "            continue\n"
                       "        # same as: python -m src.train --model <model> --seed <seed>\n"
                       '        sys.argv = ["src/train.py", "--model", model_name, "--seed", str(seed)]\n'
                       "        MAIN()")
        + script_cells("src/evaluate.py", "2. Re-evaluate a saved checkpoint",
                       "# same as: python -m src.evaluate --model small_cnn --seed 42 --split test\n"
                       '# (rewrites that prediction CSV; checkpoints are in results/checkpoints/, not committed)\n'
                       'sys.argv = ["src/evaluate.py", "--model", "small_cnn", "--seed", "42", "--split", "test"]\n'
                       'if (RESULTS / "checkpoints" / "small_cnn_seed42.pt").exists():\n'
                       "    MAIN()"))

    nbs["04_results_midterm.ipynb"] = notebook(
        "# 04. Midterm results\n\n"
        "Baseline table, per-class F1 and confusion matrices, leakage sensitivity and the Midterm questions E1-E4. "
        "Reads only the saved predictions and logs in `results/` (no model is re-run) and runs on CPU. Answers: "
        "[`docs/midterm_answers.md`](../docs/midterm_answers.md).\n\n" + GENERATED,
        script_cells("scripts/make_baseline_table.py", "1. Baseline table (part D)",
                     argv_run("scripts/make_baseline_table.py"))
        + [new_code_cell(show(tables=["midterm_baselines.csv"]))]
        + script_cells("scripts/plot_metrics.py", "2. Per-class F1 and confusion matrices (part D)",
                       argv_run("scripts/plot_metrics.py", "--seed", "42"))
        + [new_code_cell(show(["test_per_class_f1.png", "confusion_small_cnn.png", "confusion_mobilenetv3s.png",
                               "confusion_crnn.png"]))]
        + script_cells("scripts/leakage_sensitivity.py", "3. Leakage sensitivity",
                       argv_run("scripts/leakage_sensitivity.py"))
        + script_cells("scripts/midterm_questions.py", "4. Midterm questions E1-E4",
                       argv_run("scripts/midterm_questions.py", "--seed", "42"))
        + [new_code_cell(show(["midterm_e3_duration_comparison.png", "midterm_e4_tradeoff.png"],
                              ["midterm_e1_class_ranking.csv", "midterm_e3_paired_outcomes.csv",
                               "midterm_e4_tradeoff.csv"]))])
    return nbs


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, nb in build().items():
        nbformat.validate(nb)
        for c in nb.cells:
            if c.cell_type == "code":
                ast.parse(re.sub(r"(?m)^\s*[%!].*$", "", c.source))  # every code cell must be valid Python
        nbformat.write(nb, OUT / name)
        print(f"{name}: {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
