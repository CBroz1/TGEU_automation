# TGEU Automation

Automation tools to support data processing for [TGEU](tgeu.org).

## TODO

1. Edit Options files for missingness, defaulting to Uwazi version
    - [X] - Edit English options file to match Master col order
    - [X] - Edit Spanish options file to match Master col order

    - Race -> Ethnicity is the only manual editing done generally
    - Add 'sex characteristics' and 'sexual orientation' to options files
    - Default to options file for region capitalization
    - Default to Uwazi for 'unknown' vs 'unknown/not applicable'
    - Does Uwazi not yet have a Spanish-language categories loaded?
        <https://tgeu-data.uwazi.io/es/page/kbywf4gztt/trans-murder-monitoring>
    - Add new env items to example env

2. master dataset.

    - Reporting period may change, set as variable
    - 'Date added' intentionally kept blank for Uwazi
    - Any date added to Confidentiality col should be ignored
    - Pick a name format, go with it.
    - Pull both violence and unknown causes from TDoR dataset

4. Outputs

    - Generate with official asset :
    - Prevent table split across pages
    - Turn sources into numbered links to retain all

5. Yes, publishing example data is ok
6. Future

    - Where is the master dataset?
    - Attempt to merge TDoR dataset with all-time master
    - Scrub sites for frequencies of Cause of Death terms
    - From above scrub, make top and mult CoD cols

## Data Flow

- **Input Data**: The codebase relies on user-saved data, and one static file.
  - **User Saved**: In google sheets, click `File` -> `Download` ->
    `Comma-separated values (.csv)` for each of the following, and save to
    `data/`:
    - `TMM - {YEAR} - WRITE HERE - EN`, hereafter as 'master'[^1]
    - `TMM - options - EN`[^2]
    - `TMM - opciones - ES`[^2]
  - **User written**: `./data/acknowledgements.tex` is a LaTeX file with the
    acknowledgements for the yearly update pdf.
  - **Static**: The countries catalog (`./data/country_catalog.csv`) is a
        static file with English and Spanish names for countries and regions.
- **Name List**: Python converts the master list into two output files, and
    LaTeX converts the latter into a pdf:
  - `./output/{YEAR}-TMM-Namelist.csv` - to serve as the 'long' list
  - `./data/name_data_sanitized.csv` - so serve as input for the name list pdf
  - `./output/{YEAR}-TMM-Namelist.pdf` - the final pdf with each region
    and demographic subtable.[^3]
- **Yearly update**: Python uses the master list and the countries catalog to
    generate several output tables followed by the yearly update pdf:
  - `./data/yearly-region-{region}.csv` where each region is one of: Asia,
    Europe, etc.
  - `./data/yearly-demo-{demo}.csv` where demo is one of: occupation, age,
    gender, etc.
  - `./output/{YEAR}-TMM-Yearly-Update.pdf` - the final pdf

[^1]: It is assumed that any column containing the word 'privte' is a
    confidential column and should be dropped from outputs
[^2]: It is assumed that the options are listed in the same order in both
    languages, ignoring country/region columns.
[^3]: To adjust the order of these tables, run `make list` once, then edit
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
