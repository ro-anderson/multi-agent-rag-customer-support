.PHONY: clean help

SHELL=/bin/bash

## Remove Python cache files
clean:
	find . -name "__pycache__" -type d -exec rm -r {} \+

## Run Monarch chat through terminal
chat:
	poetry run python customer_support_chat/app/main.py

## Display help information
help:
	@echo "Available commands:"
	@echo "  make clean         - Remove Python cache files"
	@echo "  make chat          - Run Monarch chat through terminal"
	@echo "  make help          - Display this help information"

# Default target
.DEFAULT_GOAL := help
