# ============================================================
# Multi-Model IDS — common commands.  Run `make help` to list them.
# ============================================================
.DEFAULT_GOAL := help
.PHONY: help env lock test lint format eda tensorboard dashboard \
        docker-build docker-up docker-down clean

help:  ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

env:  ## Create the conda environment from environment.yml
	conda env create -f environment.yml

lock:  ## Freeze exact installed versions into requirements-lock.txt
	pip freeze > requirements-lock.txt

test:  ## Run the test suite
	pytest

lint:  ## Type-check and check formatting
	mypy src
	black --check src tests
	isort --check-only src tests

format:  ## Auto-format the code
	black src tests
	isort src tests

eda:  ## Run the full EDA pipeline
	python scripts/run_eda.py

tensorboard:  ## Launch TensorBoard
	tensorboard --logdir=experiments/logs/tensorboard --port=6006

dashboard:  ## Launch the Streamlit dashboard
	streamlit run dashboard/app.py

docker-build:  ## Build the Docker image
	docker compose -f docker/docker-compose.yml build

docker-up:  ## Start the container stack (lab + MLflow + TensorBoard)
	docker compose -f docker/docker-compose.yml up -d

docker-down:  ## Stop the container stack
	docker compose -f docker/docker-compose.yml down

clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info
