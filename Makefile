# ==============================================================================
# DEVELOPMENT TOOLING AND ENVIRONMENT MANAGEMENT
# ==============================================================================

.PHONY: help up down run migrate migration lint format check-types train-base train-optimized eval eval-ckpt compare test plot clean ui

# Default variables for model training/evaluation
EPOCHS ?= 50
BATCH_SIZE ?= 512
MAX_SEQ_LEN ?= 50
CHECKPOINT ?= ""
RESUME ?= 
PATIENCE ?= 10
LR ?= 1e-3
NUM_LAYERS ?= 4
NHEAD ?= 4
NUM_WORKERS ?= 8

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Docker Environment ---
up: ## Start the PostgreSQL database container
	@echo "Starting PostgreSQL container..."
	docker compose up -d

down: ## Stop the PostgreSQL database container
	@echo "Stopping PostgreSQL container..."
	docker compose down

# --- Database Migrations ---
migrate: ## Apply database migrations using Alembic
	@echo "Applying database migrations..."
	uv run alembic upgrade head

migration: ## Generate a new migration revision. Usage: make migration msg="describe changes"
	@if [ -z "$(msg)" ]; then \
		echo "Error: Please specify migration message, e.g. make migration msg='create tables'"; \
		exit 1; \
	fi
	@echo "Generating new database migration..."
	uv run alembic revision --autogenerate -m "$(msg)"

# --- Code Quality ---
lint: ## Run Ruff linter and check code styles
	@echo "Running Ruff lint checks..."
	uv run ruff check .
	@echo "Checking Ruff code formatting..."
	uv run ruff format --check .

format: ## Format Python source files in place and auto-fix linting issues
	@echo "Formatting source files..."
	uv run ruff format .
	@echo "Auto-fixing lint violations..."
	uv run ruff check --fix .

check-types: ## Check Python source files for type errors using ty
	@echo "Running type checks with ty..."
	uv run ty check app

# --- Local Execution ---
run: ## Run the local FastAPI development server
	@echo "Starting FastAPI server..."
	uv run uvicorn app.main:app --reload

train-base: ## Train the baseline Neural CAT model (Optional: EPOCHS=50 BATCH_SIZE=256 MAX_SEQ_LEN=150 RESUME=<ckpt_path> PATIENCE=10 LR=1e-3 NUM_LAYERS=4 NHEAD=4 NUM_WORKERS=8)
	PYTHONPATH=. uv run python3 scripts/train_neural_cat.py --model_type base --epochs $(EPOCHS) --batch_size $(BATCH_SIZE) --max_seq_len $(MAX_SEQ_LEN) --ckpt_path "$(RESUME)" --patience $(PATIENCE) --precision bf16-mixed --compile --lr $(LR) --num_layers $(NUM_LAYERS) --nhead $(NHEAD) --num_workers $(NUM_WORKERS)

train-optimized: ## Train the optimized Neural CAT model (Optional: EPOCHS=50 BATCH_SIZE=256 MAX_SEQ_LEN=150 RESUME=<ckpt_path> PATIENCE=10 LR=1e-3 NUM_LAYERS=4 NHEAD=4 NUM_WORKERS=8)
	PYTHONPATH=. uv run python3 scripts/train_neural_cat.py --model_type optimized --epochs $(EPOCHS) --batch_size $(BATCH_SIZE) --max_seq_len $(MAX_SEQ_LEN) --ckpt_path "$(RESUME)" --patience $(PATIENCE) --precision bf16-mixed --compile --lr $(LR) --num_layers $(NUM_LAYERS) --nhead $(NHEAD) --num_workers $(NUM_WORKERS)

eval: ## Evaluate the best checkpoint (auto-selected lowest validation loss)
	PYTHONPATH=. uv run python3 scripts/evaluate_neural_cat.py

eval-ckpt: ## Evaluate a specific checkpoint (Usage: make eval-ckpt CHECKPOINT=<path_to_ckpt>)
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: Please specify CHECKPOINT=<path_to_ckpt>"; \
		exit 1; \
	fi
	PYTHONPATH=. uv run python3 scripts/evaluate_neural_cat.py --checkpoint_path $(CHECKPOINT)

compare: ## Compare all evaluated models and update reports
	PYTHONPATH=. uv run python3 scripts/compare_models.py


plot: ## Plot training/validation loss curves from the latest log
	PYTHONPATH=. uv run python3 scripts/plot_loss.py


ui:
	PYTHONPATH=. uv run streamlit run app/infrastructure/streamlit/streamlit.py

