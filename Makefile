.PHONY: run install install-gpu dev dev-gpu clean native test release-smoke

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUNTIME_INSTALLER := $(PYTHON) scripts/install_runtime_variant.py --project . --meeting-backends none --source-venv $(VENV) --editable
export PYTHONNOUSERSITE := 1

run:
	$(PYTHON) -m nvbroadcast

install: $(VENV)
	$(RUNTIME_INSTALLER) --variant cpu

install-gpu: $(VENV)
	$(RUNTIME_INSTALLER) --variant cuda

dev: $(VENV)
	$(RUNTIME_INSTALLER) --variant cpu --development

dev-gpu: $(VENV)
	$(RUNTIME_INSTALLER) --variant cuda --development

$(VENV):
	python3 -m venv $(VENV) --system-site-packages
	$(PIP) install --upgrade pip

native:
	cd native && mkdir -p build && cd build && cmake .. && make -j$$(nproc)

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

release-smoke: $(VENV)
	$(PYTHON) scripts/release_smoke.py

clean:
	rm -rf $(VENV) build dist *.egg-info native/build
