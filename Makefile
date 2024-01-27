install:
	pip install --upgrade pip && pip install -r requirements/requirements.txt

train_pipeline:
	python bikeshare_model/train_pipeline.py

format:
	pip install -r requirements/test_requirements.txt
	black . *.py

lint:
	pylint --disable=R,C *.py

mypy:
	mypy *.py

test:
	python -m pytest tests/test_*.py

all: install format lint mypy train_pipeline test