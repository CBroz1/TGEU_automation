import logging
from datetime import datetime
from itertools import chain  # For flattening lists of lists
from pathlib import Path

import pandas as pd
from dateutil.parser import parse
from pylatexenc.latexencode import utf8tolatex

# By convention, script-level constants are in ALL_CAPS
THIS_YEAR = datetime.now().year  # Get the current year
LIMIT_FIRST_N_SOURCES = 14  # Number of sources to keep in the output
NAMES_FILE = f"./output/{THIS_YEAR}-TMM-Namelist.csv"  # Output name list
NAMES_OUTPUT_COLUMNS = {  # Columns to keep in the output CSV for name list
    "Name of the victim": "Name",
    "ReportAge": "Age",
    "Occupation": "Occupation",
    "Date of the murder dd/mm/yyyy": "Date",
    "City": "City",
    "Country/territory of the murder": "Country",
    "Type of location of the murder": "Location Type",
    "Type of murder": "Cause of death",
    "Short description (English)": "Remarks",
    "Short description (Spanish)": "Observaciones",
    "Reported by": "Reported by",
    "Link to source of information": "Sources",
}

# Logging - more informative than `print` with timestamps
logger = logging.getLogger(__name__.split(".")[0])
stream_handler = logging.StreamHandler()  # default handler
stream_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s %(levelname)-8s]: %(message)s",
        "%M:%S",
    )
)
logger.handlers = [stream_handler]


class DataPreprocessor:
    """Preprocesses input CSV file to make it LaTeX-compatible.

    Why 'class'? This allows us to split the preprocessing steps into
    smaller functions that can all reference the same DataFrame object without
    needing to pass it around as an argument. `__init__` function is run by
    default when an instance of the class is declared with 'eg = Example(args)'

    Steps:
    1. Sanitizes column names by ignoring text after a newline.
    2. Drop rows where 'Confidental' is 'Yes' or a date after the current date.
    3. Generates a 'ReportAge' column based on 'Age' and 'Age range', where
        unknown ages are replaced by the 'Age range'.
    4. Limits sources to first N items, where n in LIMIT_FIRST_N_SOURCES above.
    5. Drop unnecessary columns, only preserving OUTPUT_COLUMNS above.
    6. Remap special characaters for LaTeX

    Parameters:
    -----------
    input_file : str
        Relative path to the input CSV file.
    output_file : str
        Relative path to the output CSV file.
    debug_window : tuple
    """

    def __init__(self, input_file: str = None, debug_window: tuple = None):
        """Initialize the DataPreprocessor object.

        Parameters:
        ----------
        input_file : str, optional
            Path to the input CSV file. If none, will search for a file in the
            data directory with the current year in the filename.
        debug_window : tuple, optional
            Tuple of two integers, start and end row of input data file to use,
            for debugging issues with specific rows. If None, will process the
            entire input file.
        """
        self.df = pd.read_csv(self.fuzzy_input(input_file))
        self.names_df = None  # Initialize names_df as None, set by preproc func
        self.agg_df = None  # Initialize aggregate DataFrame as None

        if debug_window:
            start, end = debug_window
            self.df = self.df.iloc[start:end]

        _ = self.preprocess_csv()  # `_ =` says 'return val not used'
        _ = self.generate_names_list()
        _ = self.generate_aggregate()
        _ = self.save_demo_aggs()
        _ = self.save_region_aggs()
        _ = self.write_yearly_inputs_list()
        _ = self.write_total()

    def fuzzy_input(self, input_file):
        """If input file does not exist, check for file with current year."""
        input_file = input_file or "not_a_file"  # "data/example_input.csv"
        path_obj = Path(input_file)
        if path_obj.exists() and path_obj.is_file():
            return input_file

        # Defining a subfunction because this is the only place it's used

        def not_a_helper(fp):
            fn = str(fp)  # file string is not a helper csv: options, country
            banned_substrings = ["options", "opciones", "ountr", "lock"]
            if any(sub in fn for sub in banned_substrings):
                return False
            return True

        # search for first file in data labeled with current year
        input_options = [
            item
            for item in path_obj.parent.glob(f"./data/*{THIS_YEAR}*")
            if not_a_helper(item)  # exclude options files
        ]

        if input_options:
            input_file = input_options[0]
            logger.warning(f"Loading {input_file}")  # warn of change
            return input_file

        raise FileNotFoundError("Could not find input file in ./data/")

    @property  # Allows calling `DataPreprocessor.data` without parentheses
    def data(self) -> pd.DataFrame:
        return self.df  # Return the processed DataFrame

    def preprocess_csv(self):
        """Preprocess the input CSV file.

        1. Sanitizes column names by ignoring text after a newline.
        2. Generates a 'ReportAge' column based on 'Age' and 'Age range'.
        """
        # Sanitize column names: Keep text before the first line break
        self.df.columns = [col.split("\n")[0] for col in self.df.columns]
        # Warn about duplicates
        _ = self.check_duplicates()
        # Create 'ReportAge' column
        if "Age" in self.df.columns and "Age range" in self.df.columns:
            _ = self.merge_age_cols()
        else:
            logger.warning("Missing 'Age'/'Age range' columns.")
        return self.df

    def generate_names_list(self):
        """Generate a names list from the preprocessed data.

        Two outputs are generated: one as 'long' format, one LaTeX-compatible.

        1. Drop rows where 'Confidental' is 'Yes' or a future date.
        2. Limits sources to first N items, where n in LIMIT_FIRST_N_SOURCES.
        3. Drop unnecessary columns, only preserving OUTPUT_COLUMNS.
        4. Remap special characaters for LaTeX.
        """
        # Separate dataset for names list
        self.names_df = self.df.copy()  # Save a copy of the DataFrame

        # Handle 'Confidential' column filtering
        if "Confidential case" in self.names_df.columns:
            _ = self.filter_confidential()  # `_ =` says 'return val not used'
        else:
            logger.warning("Found no 'Confidential' column to filter.")

        # Drop unnecessary columns
        self.names_df = self.names_df[list(NAMES_OUTPUT_COLUMNS.keys())]
        self.names_df.rename(columns=NAMES_OUTPUT_COLUMNS, inplace=True)

        # Save as 'long' format before sanitizing content
        logger.info(f"Saving names data to {NAMES_FILE}")
        self.names_df.to_csv(NAMES_FILE, index=False)

        if "Sources" in self.names_df.columns:
            _ = self.truncate_sources()
        else:
            logger.warning("Missing Sources column.")

        # Sanitize content for LaTeX compatibility across all columns
        self.names_df = self.names_df.applymap(self.sanitize_content)
        self.names_df.to_csv("data/name_data_sanitized.csv", index=False)

        return self.names_df

    def check_duplicates(self):
        """Check for duplicates in the DataFrame."""
        check_cols = [
            col
            for col in self.df.columns  # check dupes with same name & date
            if "Name of " in col or "Date of " in col
        ]
        duplicates = self.df.duplicated(subset=check_cols, keep=False)
        if not duplicates.any():
            return
        dupes = self.df[duplicates][check_cols]
        logger.warning(f"Found duplicates name/date:\n{dupes}")

    def filter_confidential(self):
        """Drop confidential rows and columns.

        Drop columns marked as 'private': Legal name and sex assigned at birth.
        Drop rows where 'Confidential' is 'Yes' or a future date.
        """
        drop_cols = [c for c in self.df.columns if "private" in c.lower()]
        df = self.df.drop(columns=drop_cols)

        today = datetime.today().date()

        def is_confidential(value):
            """Return True if the row should be kept, False otherwise."""
            try:
                if str(value).strip().lower() == "yes":  # Check for 'Yes'
                    return False
                date_value = parse(value, fuzzy=True).date()
                return date_value <= today  # False if future date
            except (ValueError, TypeError):
                return True  # Keep wher data is not a date

        self.df = df[df["Confidential case"].apply(is_confidential)]

        return self.df

    def merge_age_cols(self):
        """Create 'ReportAge' column based on 'Age' and 'Age range'."""

        def calculate_report_age(row):
            try:  # 'try' block will catch errors and default to 'Age range'
                age = float(row["Age"])  # Check if 'Age' is a valid number
                return str(int(age))  # Return as a whole number
            except (ValueError, TypeError):  # if not valid number, use range
                return row["Age range"]

        self.df["ReportAge"] = self.df.apply(calculate_report_age, axis=1)
        return self.df

    def truncate_sources(self, n: int = LIMIT_FIRST_N_SOURCES):
        """Truncate the 'Sources' column to the first N sources.

        LaTeX will overfill the page if there are too many sources.

        Parameters:
        -----------
        source_col : str
            Name of the column containing the source links.
        n : int
            Number of sources to keep in the output.
            Default is LIMIT_FIRST_N_SOURCES.
        """

        def trunc_n_lines(content):
            """Return the first N lines of the content."""
            if not isinstance(content, str):
                return content
            return "\n".join(content.split("\n")[:n])

        self.names_df["Sources"] = self.names_df["Sources"].apply(trunc_n_lines)

        return self.df

    def replace_quotes(self, text):
        """Replaces straight double quotes with contextual LaTeX quotes.

        Assumes the text uses paired double quotes for quoting. Marches through
        input tracking if is open/close of pair. When found, replaces with LaTeX
        equivalent: `` or ''
        """
        result = []
        is_open = True  # Tracks state: current quote is open/close

        for char in text:
            if char != '"':
                result.append(char)
            else:
                ret = "``" if is_open else "''"  # LaTeX open/close
                result.append(ret)
                is_open = not is_open  # Flip, now looking for the other

        return "".join(result)

    def sanitize_content(self, x):
        """Sanitize content for LaTeX compatibility."""
        if not isinstance(x, str):
            return x

        x = x.replace("\u200B", "")  # Remove zero-width spaces
        x = x.strip("\n")  # Remove newline at the beginning and end
        x = self.replace_quotes(x)  # Replace quotes with LaTeX equiv
        x = utf8tolatex(str(x))

        content_map = {  # r"x" tells python to treat x as raw str
            ",": r"\,",  # escape commas with `\`
            "\n": r"\\",  # newline is `\\`
            "$": r"\$",  # escape `$` with `\`
        }
        for token, replaced in content_map.items():
            x = x.replace(token, replaced)
        return x

    @property
    def categories(self):
        df_en = self.load_category_df(version="en")
        df_es = self.load_category_df(version="es")

        df_en = self.add_cols(df_en, "EN")
        df_es = self.add_cols(df_es, "ES")

        df_en = df_en.melt(var_name="Category_EN", value_name="EN")
        df_es = df_es.melt(var_name="Category_ES", value_name="ES")

        # Concatenate translations, keeping both EN and ES categories
        ret = pd.concat(
            [
                df_en.reset_index(drop=True),
                df_es.reset_index(drop=True),
            ],
            axis=1,
        )
        return ret

    def add_cols(self, df, lang="EN"):
        unknown = "unknown" if lang == "EN" else "desconocido"
        orient = [
            "heterosexual",
            "queer",
            "gay/lesbian",
            "pansexual",
            unknown,
        ]
        content = {
            "EN": {
                "Sex Characteristics": ["endosex", "intersex", unknown],
                "Sexual Orientation": orient,
                "Disability": ["yes", "no", unknown],
            },
            "ES": {
                "Características sexuales": ["endosex", "intersex", unknown],
                "Orientación sexual": orient,
                "Discapacidad": ["sí", "no", unknown],
            },
        }
        for col, vals in content[lang].items():
            df[col] = None  # Initialize column with None
            df.iloc[0 : len(vals), -1] = vals  # Fill in the values
        return df

    def load_category_df(self, version: str):
        substr = "options" if version == "en" else "opciones"
        files = list(Path("data").glob(f"*{substr}*.csv"))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected 1 options file, found {files}")
        df = pd.read_csv(files[0])
        df.columns = [col.split("\n")[0] for col in df.columns]
        return df

    def countries(self):
        files = list(Path("data").glob("*ountries*catalog*.csv"))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected 1 countries file, found {files}")
        df = pd.read_csv(files[0])
        return df

    def generate_aggregate(self):
        """Generate aggregate data for summary tables."""
        df = self.df.copy()
        df.index.name = "Case"  # Set the index name for the final output
        df.reset_index(inplace=True)  # Reshape the main dataset to long reset

        # Reset column name to match options
        type_murder_col = [
            c for c in df.columns if "Type" in c and "murder" in c
        ]
        df["Type of homicide/murder"] = df[type_murder_col[0]]
        df["Race or ethnicity"] = df["Race "]

        df_long = df.melt(
            id_vars=["Case"], var_name="Category_EN", value_name="EN"
        ).dropna(subset=["EN"])

        # Merge the main dataset with categories - both english and spanish
        df_merged = pd.merge(
            df_long, self.categories, on=["Category_EN", "EN"], how="left"
        )

        # Group and aggregate to create the final output
        result = df_merged.groupby(
            ["Category_EN", "Category_ES", "EN", "ES"], as_index=False
        ).size()

        self.temp_df = df
        self.long = df_long

        self.agg_df = result.rename(columns={"size": "Count"})

        return self.agg_df

    def save_demo_aggs(self):
        # agg = self.agg_df.applymap(self.sanitize_content).copy()
        agg = self.agg_df
        table_to_col = {
            "occupation": (  # file suffix
                "Occupation",  # table column
                "Occupation/source of income",  # en header
                r"Ocupación/fuente de ingreso",  # es header
            ),
            "age": ("Age range", "Age", "Edad"),
            "migrant": (
                "Migrant status",
                "Migrant status",
                "Estato migratorio",
            ),
            "race": ("Race or ethnicity", "Race", "Raza"),
            "gender": (
                "Gender identity or expression",
                "Gender identity or expression",
                r"Identidad o expresión de género",
            ),
            "sex": (
                "Sex Characteristics",
                "Sex Characteristics",
                r"Características sexuales",
            ),
            "orientation": (
                "Sexual Orientation",
                "Sexual Orientation",
                r"Orientación sexual",
            ),
            "disablility": ("Disability", "Disability", "Discapacidad"),
            "murder": (
                "Type of homicide/murder",
                "Type of homicide/murder",
                "Tipo de homicidio/asesinato",
            ),
            "location": (
                "Type of location of the murder",
                "Type of location of homicide/murder",
                r"Tipo de ubicación del asesinato",
            ),
        }
        for table, (col, head_en, head_es) in table_to_col.items():
            sub_df = agg[agg["Category_EN"] == col]  # get the aggregated col
            if sub_df.empty:
                logging.warning(f"Problem saving {table}")
                continue
            # drop the category columns
            sub_df = sub_df.drop(columns=["Category_EN", "Category_ES"])
            sub_df = sub_df[["ES", "EN", "Count"]]  # Spanish first
            sub_df.sort_values(by="Count", ascending=False, inplace=True)
            sub_df = sub_df.rename(
                columns={"EN": head_en, "ES": head_es, "Count": " "}
            )
            sub_df.to_csv(f"data/yearly-demo-{table}.csv", index=False)

    def load_countries(self):
        data = pd.read_csv("./data/country_catalog.csv")
        return data.fillna(method="ffill")

    def save_region_aggs(self):
        all_countries = self.load_countries()

        agg_region = self.df.groupby(["Region"], as_index=False).size()
        agg_region = pd.merge(
            agg_region,
            all_countries[["Region_EN", "Region_ES"]].drop_duplicates(),
            left_on="Region",
            right_on="Region_EN",
        )

        agg_country = self.df.groupby(
            ["Country/territory of the murder"], as_index=False
        ).size()
        agg_country = pd.merge(
            agg_country,
            all_countries[["Region_EN", "Country_EN", "Country_ES"]],
            left_on="Country/territory of the murder",
            right_on="Country_EN",
        )

        for _, header in agg_region.iterrows():
            head = header.to_dict()
            countries_df = agg_country[
                agg_country["Region_EN"] == head["Region_EN"]
            ]
            countries_df = countries_df.drop(
                columns=["Country/territory of the murder", "Region_EN"]
            )
            countries_df = countries_df[["Country_ES", "Country_EN", "size"]]
            countries_df.sort_values(by="size", ascending=False, inplace=True)
            countries_df = countries_df.rename(
                columns={
                    "Country_EN": head["Region_EN"],
                    "Country_ES": head["Region_ES"],
                    "size": head["size"],
                }
            )
            fname = (
                "data/yearly-region-"
                + head["Region_EN"].replace(" ", "")
                + ".csv"
            )
            countries_df.to_csv(fname, index=False)

    def write_yearly_inputs_list(self):
        data_dir = Path("./data/")
        all_ins = chain(data_dir.glob("y*reg*csv"), data_dir.glob("y*demo*csv"))
        with_relpath = [f"./{csv}" for csv in all_ins]

        pref_order = [
            "Africa",
            "Asia",
            "Europe",
            "NorthAmerica",
            "Caribbean",
            "occupation",
            "age",
            "migrant",
            "race",
            "gender",
            "sex",
            "orientation",
            "disablility",
            "murder",
            "location",
        ]
        sorted_paths = sorted(
            with_relpath,  # sort by substrings they contain
            key=lambda x: next(
                (i for i, s in enumerate(pref_order) if s in x), len(pref_order)
            ),
        )

        with open(data_dir / "_latex_csv_inputs.txt", "w") as f:
            f.write(", \n".join(sorted_paths))

    def write_total(self):
        total = len(self.df)
        with open("./data/_total.txt", "w") as f:
            f.write(str(total))


if __name__ == "__main__":
    """Run this when called directly as `python name_list.py`."""
    pp = DataPreprocessor()
    agg = pp.agg_df
