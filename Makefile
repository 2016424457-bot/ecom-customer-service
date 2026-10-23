.PHONY: kb-build kb-reset kb-recreate kb-mine smoke-rerank ch04-eval ch04-accept eval-ch05

kb-build:
	uv run python scripts/build_knowledge.py

kb-reset:
	uv run python scripts/build_knowledge.py --reset

kb-recreate:
	uv run python scripts/build_knowledge.py --reset --recreate

kb-mine:
	uv run python scripts/mine_knowledge.py --once

smoke-rerank:
	uv run python scripts/smoke_rerank.py

ch04-eval:
	uv run python evals/run_ch04_eval.py

# 用法：make ch04-accept PHASE=build
PHASE ?= build
ch04-accept:
	uv run python scripts/ch04_acceptance.py --phase $(PHASE)

eval-ch05:
	uv run python -m scripts.eval_ch05
