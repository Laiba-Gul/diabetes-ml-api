.PHONY: install train test lint run docker-build docker-run

install:
	python -m pip install -r requirements-dev.txt

train:
	python training/train.py --download

test:
	python -m pytest --cov --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

run:
	PYTHONPATH=src uvicorn diabetes_api.main:app --reload --port 8000

docker-build:
	docker build -t diabetes-api:local .

docker-run:
	docker run --rm -p 8000:8000 --name diabetes-api diabetes-api:local
