from datetime import datetime

import pandas as pd
from dateutil.parser import parse
from pylatexenc.latexencode import utf8tolatex

LIMIT_FIRST_N_SOURCES = 14


def preprocess_csv(input_file, output_file):
    """Preprocesses input CSV file to make them LaTeX-compatible.

    1. Sanitizes column names by ignoring text after a newline.
        Removes 'dd/mm/yyyy' from 'Date of the murder' column.
    2. Drop rows where 'Confidental' is 'Yes' or a date after the current date.
    3. Generates a 'ReportAge' column based on 'Age' and 'Age range', where
        unknown ages are replaced by the 'Age range'.

    Parameters:
    -----------
    input_file : str
        Path to the input CSV file.
    output_file : str
        Path to the output CSV file with sanitized column names.
    """
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Sanitize column names: Keep text before the first line break
    df.columns = [col.split("\n")[0] for col in df.columns]

    # Handle 'Confidential' column filtering
    if "Confidential case" in df.columns:
        today = datetime.today().date()

        # Drop rows with 'Yes' or a future date in the 'Confidential' column
        def filter_confidential(value):
            try:
                # Check for 'Yes'
                if str(value).strip().lower() == "yes":
                    return False
                # Check for future date
                date_value = parse(value, fuzzy=True).date()
                return date_value <= today
            except (ValueError, TypeError):
                # Keep rows where 'Confidential' is neither 'Yes' nor a valid date
                return True

        df = df[df["Confidential case"].apply(filter_confidential)]

    # Create 'ReportAge' column
    if "Age" in df.columns and "Age range" in df.columns:

        def calculate_report_age(row):
            try:
                # Check if 'Age' is a valid number
                age = float(row["Age"])
                return str(int(age))  # Return as a whole number
            except (ValueError, TypeError):
                # Use 'Age range' if 'Age' is not a number
                return row["Age range"]

        df["ReportAge"] = df.apply(calculate_report_age, axis=1)

    # Only keep the required columns
    cols = [
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
        "Link to source of information",
    ]
    df = df[cols]

    def truncate_n_lines(content, n=LIMIT_FIRST_N_SOURCES):
        if not isinstance(content, str):
            return content
        return "\n".join(content.split("\n")[:n])

    source_col = "Link to source of information"
    if source_col in df.columns:
        df["FirstNSources"] = df[source_col].apply(truncate_n_lines)

    df = df.iloc[6:8]  # debug first half

    # Replace content for latex:
    content_map = {
        ",": r"\,",
        "\n": r"\\",
        "$": r"\$",
    }

    def replace_quotes(text):
        """
        Replaces straight double quotes with contextual LaTeX quotes.
        Assumes the text uses paired double quotes for quoting.
        """
        result = []
        is_open = True  # Tracks whether the current quote is opening or closing

        for char in text:
            if char == '"':
                if is_open:
                    result.append("``")  # LaTeX opening quotes
                else:
                    result.append("''")  # LaTeX closing quotes
                is_open = not is_open
            else:
                result.append(char)

        return "".join(result)

    def sanitize_content(x):
        if not isinstance(x, str):
            return x
        x = x.replace("\u200B", "")  # Remove zero-width spaces
        x = x.strip("\n")  # Remove newline at the beginning and end
        x = replace_quotes(x)  # Replace straight quotes with LaTeX quotes
        # split_x = x.split("\n")
        # if len(split_x) > 11:
        #     print(len(split_x))
        #     x = "\n".join(split_x[:14])
        x = utf8tolatex(str(x))
        # x = utf8tolatex(str(x))
        for token, replaced in content_map.items():
            x = x.replace(token, replaced)
        return x

    df = df.applymap(sanitize_content)

    # Save the processed CSV
    df.to_csv(output_file, index=False)
    return df


input_csv = "./data/name_data.csv"
output_csv = "./data/name_data_sanitized.csv"
df = preprocess_csv(input_csv, output_csv)
