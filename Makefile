.PHONY: install run demo sim

install:
	pip install fastapi "uvicorn[standard]" pydantic

run:
	uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload

demo:
	python sim/load_sim.py 5

sim:
	python sim/load_sim.py 20
