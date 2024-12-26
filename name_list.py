import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from dateutil.parser import parse
from pylatexenc.latexencode import utf8tolatex

# By convention, script-level constants are in ALL_CAPS
LIMIT_FIRST_N_SOURCES = 14  # Number of sources to keep in the output
OUTPUT_COLUMNS = [  # Columns to keep in the output CSV for name list
    "Name of the victim",
    "ReportAge",
    "Occupation",
    "Date of the murder dd/mm/yyyy",
    "City",
    "Country/territory of the murder",
    "Type of location of the murder",
    "Type of murder",
    "Short description (English)",
    "Short description (Spanish)",
    "Reported by",
    "FirstNSources",
]

logger = logging.getLogger(__name__)  # more specific output than 'print'


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

    def __init__(
        self, input_file: str, output_file=str, debug_window: tuple = None
    ):
        """Initialize the DataPreprocessor object."""
        self.df = pd.read_csv(self.fuzzy_input(input_file))
        if debug_window:
            start, end = debug_window
            self.df = self.df.iloc[start:end]

        _ = self.preprocess_csv()  # `_ =` says 'return val not used'

        self.df.to_csv(output_file, index=False)

    def fuzzy_input(self, input_file):
        """If input file does not exist, check for file with current year."""
        path_obj = Path(input_file)
        if not path_obj.exists():
            year = datetime.now().year  # get this year
            # search for first file in data labeled with current year
            input_file = str(next(path_obj.parent.glob(f"*{year}*")))
            logger.warning(f"Loading data from {input_file}")  # warn of change
        return input_file

    @property  # Allows calling `DataPreprocessor.data` without parentheses
    def data(self) -> pd.DataFrame:
        return self.df  # Return the processed DataFrame

    def preprocess_csv(self):
        """Preprocess the input CSV file to make it LaTeX-compatible.

        1. Sanitizes column names by ignoring text after a newline.
        2. Drop rows where 'Confidental' is 'Yes' or a future date.
        3. Generates a 'ReportAge' column based on 'Age' and 'Age range'.
        4. Limits sources to first N items, where n in LIMIT_FIRST_N_SOURCES.
        5. Drop unnecessary columns, only preserving OUTPUT_COLUMNS.
        6. Remap special characaters for LaTeX.
        """

        # Sanitize column names: Keep text before the first line break
        self.df.columns = [col.split("\n")[0] for col in self.df.columns]

        # Handle 'Confidential' column filtering
        if "Confidential case" in self.df.columns:
            _ = self.filter_confidential()  # `_ =` says 'return val not used'
        else:
            logger.warning("Found to 'Confidential' column to filter.")

        # Create 'ReportAge' column
        if "Age" in self.df.columns and "Age range" in self.df.columns:
            _ = self.merge_age_cols()
        else:
            logger.warning("Missing 'Age'/'Age range' columns.")

        source_col = "Link to source of information"
        if source_col in self.df.columns:
            _ = self.truncate_sources(source_col)
        else:
            logger.warning(f"Missing '{source_col}' column.")

        # Drop unnecessary columns
        self.df = self.df[OUTPUT_COLUMNS]

        # Sanitize content for LaTeX compatibility across all columns
        self.df = self.df.applymap(self.sanitize_content)

        return self.df

    def filter_confidential(self):
        """Drop rows where 'Confidential' is 'Yes' or a future date."""
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

        self.df = self.df[self.df["Confidential case"].apply(is_confidential)]
        return self.df

    def merge_age_cols(self):
        """Create 'ReportAge' column based on 'Age' and 'Age range'."""

        def calculate_report_age(row):
            try:
                age = float(row["Age"])  # Check if 'Age' is a valid number
                return str(int(age))  # Return as a whole number
            except (ValueError, TypeError):  # if not valid number, use range
                return row["Age range"]

        self.df["ReportAge"] = self.df.apply(calculate_report_age, axis=1)
        return self.df

    def truncate_sources(self, source_col: str, n: int = LIMIT_FIRST_N_SOURCES):
        """Truncate the 'Link to source' column to the first N sources.

        LaTeX will overfill the page if there are too many sources.

        Parameters:
        -----------
        source_col : str
            Name of the column containing the source links.
        n : int
            Number of sources to keep in the output.
            Default is LIMIT_FIRST_N_SOURCES.
        """

        def truncate_n_lines(content):
            if not isinstance(content, str):
                return content
            return "\n".join(content.split("\n")[:n])

        self.df["FirstNSources"] = self.df[source_col].apply(truncate_n_lines)

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


if __name__ == "__main__":
    """Run this when called directly as `python name_list.py`."""
    input_csv = "./data/name_data.csv"
    output_csv = "./data/name_data_sanitized.csv"
    preprocessor = DataPreprocessor(input_csv, output_csv, debug_window=(6, 8))
    df = DataPreprocessor.data
