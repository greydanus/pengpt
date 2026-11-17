#!/bin/bash
cd "$(dirname "$0")/.." && ${PY:-python} paper/figures/overview.py && ${PY:-python} paper/tables.py > /dev/null && cd paper && tectonic pengpt.tex
