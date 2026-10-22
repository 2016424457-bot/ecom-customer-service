.PHONY: kb-build kb-reset

kb-build:
	uv run python scripts/build_knowledge.py

kb-reset:
	uv run python scripts/build_knowledge.py --reset
