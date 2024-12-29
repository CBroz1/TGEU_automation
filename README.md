# TGEU Automation

Automation tools to support data processing for [TGEU](tgeu.org).

## Data Flow

- **Input Data**: In google sheets, click `File` -> `Download` ->
    `Comma-separated values (.csv)` for each of the following, and save to
    `data/`:
  - `TMM - {YEAR} - WRITE HERE - EN`, hereafter as 'master'
  - `tmm-main-file ... Countries catalog`
  - `TMM - options - EN`[^1]
  - `TMM - opciones - ES`[^1]
- **Name List**: Python converts the master list into two output files, and
    LaTeX converts the latter into a pdf:
  - `./output/{YEAR}-TMM-Namelist.csv` - to serve as the 'long' list
  - `./data/name_data_sanitized.csv` - so serve as input for the name list pdf
  - `./output/{YEAR}-TMM-Namelist.pdf` - the final pdf
- **Yearly update**: Python uses the master list and the countries catalog to
    generate several output tables followed by the yearly update pdf:
  - `./data/yearly-region-{region}.csv` where each region is one of: Asia,
    Europe, etc.
  - `./data/yearly-demo-{demo}.csv` where demo is one of: occupation, age,
    gender, etc.
  - `./output/{YEAR}-TMM-Yearly-Update.pdf` - the final pdf

[^1]: It is assumed that the options are listed in the same order in both
    languages, ignoring country/region columns.

## Name list csv to pdf

TODO:

- python:
  - Why does the total differ?
- tex
  - prevent page breaks in the middle of tables
  - move acknowledgements to seperate file
- organization
  - tex work to subdirectory
  - makefile? python main?
  - tex namelist should move to `./output/{YEAR}-TMM-Namelist.pdf`
- env file for user config
  - latex command
  - year - assume from input file name?
- questions
  - confirm confidentiality date after today assumption
  - confirm source count limit in pdf output
  - ok to publish example data?
  - logo variant with text
  - tmm-main-file-TDOR2024:TMM - options - EN:P-Q had a repeat column name
    that interferes with exporting the CSV. Previously P was empty. I copied
    the data over from Q to P.
  - I had to manually add 'Sex characteristics' and '"Sexual orientation"' to
    the options file.
  - How likely is it for the options file(s) to change?
  - My first draft assumed options files were in the same order across
    languages, but this wasn't the case for coutries/regions.
    Is this a safe assumption for other columns?
  - Would we be able to modify the headings on the options files to match the
    headings on the yearly report?
  - in the yearly update 'unknown' is often converted to
    'unknown / not applicable' - is this intentional? should this be done in a
    pdf i produce?
  - I noticed inconsistent capitalization of region/subregion names in Spanish. Is that intentional?
  - Is your reporting period always Oct 31 - Sept 30? I wanted to use the
    'Date added' column, but it was empty
  - Is the order of the tables in the yearly update important?
  - Are we okay with page breaks in the middle of tables?
