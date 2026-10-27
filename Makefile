.PHONY: install test lint fmt typecheck run up down
install:
	uv sync
test:
	uv run pytest
lint:
	uv run ruff check . && uv run ruff format --check .
fmt:
	uv run ruff check --fix . && uv run ruff format .
typecheck:
	uv run mypy
run:
	uv run uvicorn ragkit.api.main:app --reload
up:
	docker compose up --build
down:
	docker compose down
