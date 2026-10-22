.PHONY: kb-build kb-reset kb-mine

kb-build:
	uv run python scripts/build_knowledge.py

kb-reset:
	uv run python scripts/build_knowledge.py --reset

kb-mine:
	uv run python scripts/mine_knowledge.py --once
