.PHONY: setup lint format test grobid-up grobid-down fetch parse extract eval summarize compare novelty novelty-eval methods methods-eval all

UV ?= uv
RUN := $(UV) run

setup:
	$(UV) sync --all-extras
	$(RUN) pre-commit install
	@test -f .env || cp .env.example .env

lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format:
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

test:
	$(RUN) pytest

grobid-up:
	docker compose up -d grobid
	@echo "waiting for GROBID..."
	@for i in $$(seq 1 90); do curl -sf http://localhost:8070/api/isalive >/dev/null && echo "GROBID up" && exit 0; sleep 2; done; echo "GROBID did not come up"; exit 1

grobid-down:
	docker compose down

fetch:
	$(RUN) reviewer fetch

parse:
	$(RUN) reviewer parse data/papers

extract:
	$(RUN) reviewer extract

eval:
	$(RUN) reviewer eval

novelty:
	$(RUN) reviewer novelty check

novelty-eval:
	$(RUN) reviewer novelty eval --split dev

methods:
	$(RUN) reviewer methods check

methods-eval:
	$(RUN) reviewer methods eval

summarize:
	$(RUN) reviewer eval summarize

compare:
	$(RUN) reviewer compare

all: fetch parse extract eval

