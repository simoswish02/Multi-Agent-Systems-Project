# Build automation for the Multi-Agent Systems project.
#
#   make install      create the conda environment (or: make install-pip)
#   make train        curriculum training from configs/default.yaml
#   make eval         render one greedy episode with the GUI
#   make simulate     GUI setup screen (fault slider, auto-nav toggle)
#   make experiments  the four evaluation sweeps behind Section 7
#   make figures      regenerate every figure of the report
#   make report       compile report/Rimondi-MultiDroneSearch-MAS.pdf
#   make all          figures + report
#   make clean        remove LaTeX build artefacts and __pycache__
#
# Variables can be overridden on the command line, e.g.
#   make eval CKPT=checkpoints/best.pt SEEDS=100

PYTHON ?= python
CONFIG ?= configs/default.yaml
CKPT   ?= checkpoints/mas_50k_dr_faults_ep50000.pt
SEEDS  ?= 500
HSEEDS ?= 300

.PHONY: all install install-pip train resume eval simulate play experiments \
        figures report clean help

all: figures report

# --- environment -----------------------------------------------------------

install:
	conda env create -f environment.yaml

install-pip:
	$(PYTHON) -m pip install -r requirements.txt

# --- running ---------------------------------------------------------------

train:
	$(PYTHON) main.py --mode train --config $(CONFIG)

resume:
	$(PYTHON) main.py --mode train --config $(CONFIG) --resume checkpoints/last.pt

eval:
	$(PYTHON) main.py --mode eval --config $(CONFIG) --checkpoint $(CKPT)

simulate:
	$(PYTHON) main.py --mode simulate --config $(CONFIG)

play:
	$(PYTHON) main.py --mode play --config $(CONFIG)

# --- results ---------------------------------------------------------------
# Writes testing_results/csv/. Hours of GPU time; see hpc/ for the SLURM jobs.

experiments:
	$(PYTHON) testing/evaluate_policy.py --axis all      --seeds $(SEEDS)  --checkpoint $(CKPT) --out-dir testing_results
	$(PYTHON) testing/evaluate_policy.py --axis fault    --seeds $(SEEDS)  --checkpoint $(CKPT) --out-dir testing_results/auto_nav --auto-nav
	$(PYTHON) testing/evaluate_policy.py --axis heatmaps --seeds $(HSEEDS) --checkpoint $(CKPT) --out-dir testing_results
	$(PYTHON) testing/evaluate_policy.py --axis epoch    --seeds $(HSEEDS) --ckpt-glob "checkpoints/mas_50k_dr_faults_ep*.pt" --out-dir testing_results

# Every figure in the report: the notebook owns the ones derived from the
# evaluation CSVs, make_training_figures.py the two training-side ones.
figures:
	jupyter nbconvert --to notebook --execute --inplace testing/analysis.ipynb
	$(PYTHON) testing/make_training_figures.py

# --- report ----------------------------------------------------------------
# latexmk is not used: it needs perl, which the authoring machine lacks.

report:
	cd report && pdflatex -interaction=nonstopmode Rimondi-MultiDroneSearch-MAS.tex \
	  && bibtex Rimondi-MultiDroneSearch-MAS \
	  && pdflatex -interaction=nonstopmode Rimondi-MultiDroneSearch-MAS.tex \
	  && pdflatex -interaction=nonstopmode Rimondi-MultiDroneSearch-MAS.tex

clean:
	rm -f report/*.aux report/*.bbl report/*.blg report/*.log \
	      report/*.out report/*.toc report/sections/*.aux
	rm -rf __pycache__ */__pycache__ */*/__pycache__

help:
	@sed -n '3,16p' Makefile
