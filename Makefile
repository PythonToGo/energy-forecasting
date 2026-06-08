.PHONY: install lint format typecheck test verify preprocess train clean

install:        ## Install runtime + dev dependencies (editable)
	uv pip install -e ".[dev]"

lint:           ## Lint with ruff
	ruff check src/

format:         ## Auto-format with ruff
	ruff format src/

typecheck:      ## Static type-check (strict)
	mypy src/ --strict

test:           ## Run unit tests with coverage
	pytest

verify: lint typecheck test   ## Run the full local quality gate

preprocess:     ## Build the cleaned dataset
	python src/data_loader.py

train:          ## Train the model and log to MLflow
	python src/train_model.py

clean:          ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ruff_cache .mypy_cache .pytest_cache
