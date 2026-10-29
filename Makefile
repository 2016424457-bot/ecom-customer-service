.PHONY: kb-build kb-reset kb-recreate kb-mine smoke-rerank ch04-eval ch04-accept eval-ch05 eval-ch06-prompts eval-ch06 eval-ch07 mcp-logistics mcp-aftersales

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

eval-ch06-prompts:
	uv run python -m scripts.eval_ch06 --prompts

eval-ch06:
	uv run python -m scripts.eval_ch06 --e2e

eval-ch07:
	uv run python -m scripts.eval_ch07

eval-ch08:
	uv run python -m scripts.eval_ch08

mcp-logistics:
	uv run python -m mcp_servers.logistics_server

mcp-aftersales:
	uv run python -m mcp_servers.aftersales_server
