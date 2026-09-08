.PHONY: test baseline status exercises check-cpu

test:
	python -m pytest -q

baseline:
	python -m megatron_mini.train --mode baseline --steps 20

status:
	python -m megatron_mini.tasks

exercises:
	python -m pytest -q --exercises

check-cpu:
	bash scripts/check_cpu.sh

.PHONY: labs check-v2
labs:
	python -m megatron_mini.lab --list

check-v2:
	bash scripts/check_v2_cpu.sh
