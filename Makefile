# Makefile

.PHONY: all enhance check quality

# Default target
all: enhance quality check

# Run image enhancement
inpainting:
	@echo "Running image enhancement..."
	python3 src/inpainting.py

# Run quality enhancement
quality:
	@echo "Running quality enhancement..."
	python3 src/quality_enhancement.py

# Run consistency check
check:
	@echo "Running consistency check..."
	python3 src/check_consistency.py

