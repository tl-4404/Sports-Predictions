# SportsAid — one-command setup + daily driver.
#
# First run:   make setup     (venv + deps + .env scaffold + preflight)
# Then:        edit .env with your keys
#              make doctor     (confirm what's live-ready)
#              make run        (full pipeline)   &&   make web  (the board)
#
# Everything runs in a local .venv so it never touches system Python.

.DEFAULT_GOAL := help
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install
install: $(VENV)                       ## install serving/live deps
	$(PIP) install -r requirements.txt

.PHONY: install-training
install-training: $(VENV)              ## heavyweight deps (torch/pybaseball) for caches + NBA
	$(PIP) install -r requirements-training.txt

.PHONY: env
env:                                   ## scaffold .env from .env.example (never overwrites)
	@test -f .env && echo ".env already exists — leaving it untouched" \
		|| (cp .env.example .env && echo "created .env — fill in your keys")

.PHONY: setup
setup: install env doctor              ## one-command first run: deps + .env + preflight
	@echo ""
	@echo "Setup done. Edit .env (ANTHROPIC_API_KEY at minimum), then:  make run  &&  make web"

.PHONY: doctor
doctor: $(VENV)                        ## preflight: what's ready for a live run
	$(PY) run_pipeline.py --stage doctor

.PHONY: run
run: $(VENV)                           ## full pipeline: odds -> predict -> llm -> prizepicks -> grade
	$(PY) run_pipeline.py

.PHONY: odds predict llm prizepicks grade
odds predict llm prizepicks grade: $(VENV)   ## run a single pipeline stage
	$(PY) run_pipeline.py --stage $@

.PHONY: web
web: $(VENV)                           ## Flask dev server (the board at :5000)
	$(PY) app.py

.PHONY: backtest
backtest: $(VENV)                      ## real historical check: make backtest DATASET=... ODDS=...
	@test -n "$(DATASET)" || (echo "DATASET is required; synthetic is only a self-test" && exit 2)
	@test -n "$(ODDS)" || (echo "ODDS archive is required for market-relative validation" && exit 2)
	$(PY) mlb/backtest.py "$(DATASET)" --odds "$(ODDS)"

.PHONY: selftest
selftest: $(VENV)                      ## simulator engine self-consistency only (not accuracy evidence)
	$(PY) mlb/backtest.py --synthetic --flat-market

.PHONY: test
test: $(VENV)                          ## run all offline test suites (no network)
	$(PY) tests/test_pipeline_offline.py
	$(PY) tests/test_prizepicks_offline.py
	$(PY) tests/test_interface_and_toggle.py
	$(PY) tests/test_infra.py
	$(PY) tests/test_app_offline.py
	$(PY) tests/test_llm_chat_pipeline.py
	$(PY) tests/test_mlb_audit_regressions.py
	$(PY) tests/test_prizepicks_fetcher_audit.py
	$(PY) tests/test_safety_regressions.py
	$(PY) tests/test_accuracy_policy.py
	$(PY) tests/test_merged_audit_policy.py

.PHONY: clean
clean:                                 ## remove the venv
	rm -rf $(VENV)

.PHONY: help
help:                                  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
