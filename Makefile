# ==============================================================================
# DEVELOPMENT TOOLING AND ENVIRONMENT MANAGEMENT
# ==============================================================================

.PHONY: help up down run migrate migration lint format check-types

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

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
