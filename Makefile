.PHONY: test demo lint
test:
	PYTHONPATH=src python3 -m pytest -q
demo:
	python3 demo.py
lint:
	python3 -m compileall -q src demo.py
