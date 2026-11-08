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
	-$(PY) mlb/pipeline.py --stage doctor

.PHONY: run
run: $(VENV)                           ## full pipeline: odds -> predict -> llm -> prizepicks -> grade
	$(PY) mlb/pipeline.py

.PHONY: odds predict llm prizepicks grade
odds predict llm prizepicks grade: $(VENV)   ## run a single pipeline stage
	$(PY) mlb/pipeline.py --stage $@

.PHONY: web
web: $(VENV)                           ## Flask dev server (the board at :5000)
	$(PY) shared/web/app.py

.PHONY: backtest
backtest: $(VENV)                      ## synthetic calibration + flat-market ROI check
	$(PY) -m mlb.models.training.backtest --synthetic --flat-market

.PHONY: backtest-real
backtest-real: $(VENV)                 ## run real dataset: make backtest-real DATA=... SIMS=10000
	@test -n "$(DATA)" || (echo "DATA=path/to/backtest.json is required" && exit 2)
	$(PY) -m mlb.models.training.backtest $(DATA) --sims $(or $(SIMS),10000)

.PHONY: backtest-data
backtest-data: $(VENV)                 ## build data: make backtest-data START=... END=... WARMUP=... OUT=...
	@test -n "$(START)" || (echo "START=YYYY-MM-DD is required" && exit 2)
	@test -n "$(END)" || (echo "END=YYYY-MM-DD is required" && exit 2)
	@test -n "$(WARMUP)" || (echo "WARMUP=YYYY-MM-DD is required" && exit 2)
	$(PY) -m mlb.data_fetching.build_backtest_data $(START) $(END) \
		--warmup-start $(WARMUP) --out $(or $(OUT),mlb/data/cache/backtest.json)

.PHONY: install-dev
install-dev: $(VENV)                   ## install serving + offline test dependencies
	$(PIP) install -r requirements-dev.txt

.PHONY: test test-nba test-mlb test-shared test-web
test: $(VENV)                          ## run every offline suite in an isolated process
	$(PY) scripts/run_tests.py

test-nba test-mlb test-shared test-web: $(VENV) ## run tests owned by one component
	$(PY) scripts/run_tests.py --suite $(patsubst test-%,%,$@)

.PHONY: clean
clean:                                 ## remove the venv
	rm -rf $(VENV)

.PHONY: help
help:                                  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
