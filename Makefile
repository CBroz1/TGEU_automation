# Variables
LATEX_CMD := $(shell grep ^LATEX_CMD .env | cut -d '=' -f2)
THIS_YEAR := $(shell grep ^THIS_YEAR .env | cut -d '=' -f2)
SRC_DIR := ./src
OUTPUT_DIR := ./output
PROCESS_SCRIPT := $(SRC_DIR)/process.py
TOTAL_FILE := ./data/_total.txt

# Targets
.PHONY: list report both clean

list: $(OUTPUT_DIR)/$(THIS_YEAR)-TMM-Namelist.pdf

report: $(OUTPUT_DIR)/$(THIS_YEAR)-TMM-Report.pdf

both: list report clean

$(TOTAL_FILE):
	@echo "Checking for required data file..."
	@if [ ! -f $(TOTAL_FILE) ]; then \\
		echo "Data file not found, running preprocessing..."; \\
		python $(PROCESS_SCRIPT); \\
	fi

$(OUTPUT_DIR)/$(THIS_YEAR)-TMM-Namelist.pdf: $(SRC_DIR)/name_list.tex $(TOTAL_FILE)
	@echo "Compiling Namelist PDF..."
	@$(LATEX_CMD) -interaction=batchmode -output-directory=$(SRC_DIR) \
		$(SRC_DIR)/name_list.tex >/dev/null
	@mv $(SRC_DIR)/name_list.pdf $(OUTPUT_DIR)/$(THIS_YEAR)-TMM-Namelist.pdf

$(OUTPUT_DIR)/$(THIS_YEAR)-TMM-Report.pdf: $(SRC_DIR)/yearly_report.tex $(TOTAL_FILE)
	@echo "Compiling Report PDF..."
	@$(LATEX_CMD) -interaction=batchmode -output-directory=$(SRC_DIR) \
		$(SRC_DIR)/yearly_report.tex >/dev/null
	@mv $(SRC_DIR)/yearly_report.pdf $(OUTPUT_DIR)/$(THIS_YEAR)-TMM-Report.pdf

clean:
	@echo "Cleaning up..."
	@rm -f $(SRC_DIR)/*.aux $(SRC_DIR)/*.log $(SRC_DIR)/*.pdf $(SRC_DIR)/*.out
	@rm -f $(TOTAL_FILE)

