.PHONY: install lint format typecheck test check build docker-build corpus-build corpus-verify seed-extract eval eval-retrieval eval-prepare eval-submit-generation eval-status-generation eval-download-generation eval-prepare-topup eval-submit-generation-topup eval-status-generation-topup eval-download-generation-topup eval-submit-review eval-status-review eval-download-review eval-submit-review-topup eval-status-review-topup eval-download-review-topup eval-queue eval-apply-notes eval-review eval-promote up down

install:
	cd backend && uv sync --extra dev
	cd frontend && npm ci

lint:
	cd backend && uv run ruff check app tests scripts eval
	cd backend && uv run ruff format --check app tests scripts eval
	cd frontend && npm run lint

format:
	cd backend && uv run ruff format app tests scripts eval
	cd backend && uv run ruff check --fix app tests scripts eval

typecheck:
	cd frontend && npm run typecheck

test:
	cd backend && uv run pytest --cov=app --cov-report=term --cov-report=xml --cov-fail-under=85
	cd backend && uv run coverage report --include='app/rag/*' --fail-under=90
	cd frontend && npm run test:coverage

check: lint typecheck test build corpus-verify

build:
	cd frontend && npm run build

docker-build:
	docker build --tag tanya-lalin:local .

corpus-build:
	cd backend && uv run python -m scripts.build_corpus

corpus-verify:
	cd backend && uv run python -m scripts.verify_corpus

seed-extract:
	cd backend && uv run --extra data python -m scripts.extract_hukumonline_seed

eval:
	cd backend && uv run python -m eval.run_eval

eval-retrieval:
	cd backend && uv run python -m eval.retrieval_experiment

eval-prepare:
	cd backend && uv run --extra data python -m eval.dataset_pipeline prepare

eval-submit-generation:
	cd backend && uv run --extra data python -m eval.dataset_pipeline submit-generation

eval-status-generation:
	cd backend && uv run --extra data python -m eval.dataset_pipeline status-generation

eval-download-generation:
	cd backend && uv run --extra data python -m eval.dataset_pipeline download-generation

eval-prepare-topup:
	cd backend && uv run --extra data python -m eval.dataset_pipeline prepare-topup

eval-submit-generation-topup:
	cd backend && uv run --extra data python -m eval.dataset_pipeline submit-generation-topup

eval-status-generation-topup:
	cd backend && uv run --extra data python -m eval.dataset_pipeline status-generation-topup

eval-download-generation-topup:
	cd backend && uv run --extra data python -m eval.dataset_pipeline download-generation-topup

eval-submit-review:
	cd backend && uv run --extra data python -m eval.dataset_pipeline submit-review

eval-status-review:
	cd backend && uv run --extra data python -m eval.dataset_pipeline status-review

eval-download-review:
	cd backend && uv run --extra data python -m eval.dataset_pipeline download-review

eval-submit-review-topup:
	cd backend && uv run --extra data python -m eval.dataset_pipeline submit-review-topup

eval-status-review-topup:
	cd backend && uv run --extra data python -m eval.dataset_pipeline status-review-topup

eval-download-review-topup:
	cd backend && uv run --extra data python -m eval.dataset_pipeline download-review-topup

eval-queue:
	cd backend && uv run --extra data python -m eval.dataset_pipeline queue

eval-apply-notes:
	cd backend && uv run --extra data python -m eval.dataset_pipeline apply-notes

eval-review:
	cd backend && uv run --extra data python -m eval.dataset_pipeline review

eval-promote:
	cd backend && uv run --extra data python -m eval.dataset_pipeline promote --confirm-legal-review --confirm-licensing

up:
	docker compose up -d --build

down:
	docker compose down --remove-orphans
