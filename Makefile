install:
	pip install --upgrade pip && pip install -r requirements/requirements.txt && pip install -r requirements/test_requirements.txt

train_pipeline:
	python bikeshare_model/train_pipeline.py

format:
	black . *.py

lint:
	pylint --disable=R,C *.py

mypy:
	mypy --install-types bikeshare_model/ tests/

test:
	python -m pytest tests/test_*.py

all: install format lint train_pipeline test