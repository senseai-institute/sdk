# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
.PHONY: test lint type fmt bench

test:
	uv run pytest -m "not slow" --cov --cov-report=term-missing:skip-covered

test-all:
	uv run pytest --cov --cov-report=term-missing:skip-covered

lint:
	uv run ruff check src tests benchmarks examples
	uv run ruff format --check src tests benchmarks examples

fmt:
	uv run ruff format src tests benchmarks examples
	uv run ruff check --fix src tests benchmarks examples

type:
	uv run mypy

bench:
	uv run python benchmarks/bench_pipeline.py
