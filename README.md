# TGEU Automation

Automation tools to support data processing for [TGEU](tgeu.org).

## Data Flow

- **Input Data**: The codebase relies on user-saved data, and one static file.
  - **User Saved**: In google sheets, click `File` -> `Download` ->
    `Comma-separated values (.csv)` for each of the following, and save to
    `data/`:
    - `TMM - {YEAR} - WRITE HERE - EN`, hereafter as 'master'
    - `TMM - options - EN`[^1]
    - `TMM - opciones - ES`[^1]
  - **User written**: `./data/acknowledgements.tex` is a LaTeX file with the
    acknowledgements for the yearly update pdf.
  - **Static**: The countries catalog (`./data/country_catalog.csv`) is a
        static file with English and Spanish names for countries and regions.
- **Name List**: Python converts the master list into two output files, and
    LaTeX converts the latter into a pdf:
  - `./output/{YEAR}-TMM-Namelist.csv` - to serve as the 'long' list
  - `./data/name_data_sanitized.csv` - so serve as input for the name list pdf
  - `./output/{YEAR}-TMM-Namelist.pdf` - the final pdf with each region
    and demographic subtable.[^2]
- **Yearly update**: Python uses the master list and the countries catalog to
    generate several output tables followed by the yearly update pdf:
  - `./data/yearly-region-{region}.csv` where each region is one of: Asia,
    Europe, etc.
  - `./data/yearly-demo-{demo}.csv` where demo is one of: occupation, age,
    gender, etc.
  - `./output/{YEAR}-TMM-Yearly-Update.pdf` - the final pdf

[^1]: It is assumed that the options are listed in the same order in both
    languages, ignoring country/region columns.
[^2]: To adjust the order of these tables, run `make list` once, then edit
    `./data/_latex_csv_order.csv` to your liking, and run `make list` again.

## Setup

- **Python**: The codebase is written in Python 3.9.2. Install the required
    packages with `pip install -r requirements.txt`.
- **LaTeX**: The codebase uses `xelatex` to convert the pdfs, but any LaTeX
  compiler should work.
- **env**: You'll need to copy `example.env` to `.env` and fill in the
    appropriate values, including
  - your API key for tdor.translivesmatter.info
  - your LaTeX compiler command
  - the year for the data you're processing
- **Running**: The `Makefile` will allow you to run the entire process with
    `make both`, or `make list` and `make report` for the individual steps.
