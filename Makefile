.PHONY: run install dev clean native test release-smoke

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

run:
	$(PYTHON) -m nvbroadcast

install: $(VENV)
	$(PIP) install -e .

dev: $(VENV)
	$(PIP) install -e ".[dev]"

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
