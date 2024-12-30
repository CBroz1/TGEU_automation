import random
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.parser import ParserError, parse
from faker import Faker

# Initialize the Faker instance
fake = Faker()
fake_es = Faker("es_ES")


def randomize_rows(df):
    shuffled_df = df.copy()
    for col in shuffled_df.columns:
        shuffled_df[col] = np.random.permutation(shuffled_df[col].values)
    return shuffled_df


def date_parse(date):
    try:
        return parse(date, fuzzy=True)
    except (ValueError, TypeError, ParserError):
        return None


def generate_fake_data(column_name, original_column):
    """Generate fake data for a given column."""
    if original_column.dtype == "int64" or original_column.dtype == "float64":
        # Numeric columns: Random numbers in the range of the original data
        min_value = original_column.min()
        max_value = original_column.max()
        return [
            (
                random.uniform(min_value, max_value)
                if original_column.dtype == "float64"
                else random.randint(min_value, max_value)
            )
            for _ in range(len(original_column))
        ]
    elif (
        "name" in column_name.lower()
        and "perpetrator" not in column_name.lower()
    ):
        # Generate fake names, but preserve 'unknown'
        return [
            value if str(value).lower() == "unknown" else fake.name()
            for value in original_column
        ]
    elif "date" in column_name.lower() and "Disability" not in column_name:
        # Generate fake dates
        only_dates = original_column.apply(date_parse).dropna()
        min_date = only_dates.min()
        max_date = only_dates.max()
        return [
            (
                value
                if str(value).lower() == "unknown"
                else fake.date_between(start_date=min_date, end_date=max_date)
            )
            for value in original_column
        ]
    elif (
        "description" in column_name.lower()
        or "clarification" in column_name.lower()
        or "notes" in column_name.lower()
        and "attachment" not in column_name.lower()
    ):
        # Description columns: Generate fake text with n words
        word_counts = original_column.str.split().str.len()
        min_count = word_counts.min()
        max_count = word_counts.max()
        random_count = random.randint(min_count, max_count)
        this_fake = fake if "english" in column_name.lower() else fake_es
        return [
            " ".join(this_fake.words(random_count))
            for _ in range(len(original_column))
        ]
    elif "source" in column_name.lower():
        # Sources columns: Generate fake URLs with n linebreaks
        linebreak_count = original_column.str.count("\n")
        min_breaks = linebreak_count.min()
        max_breaks = min(linebreak_count.max(), 5)
        return [
            "\n".join(
                [
                    fake_es.url()
                    for _ in range(random.randint(min_breaks, max_breaks))
                ]
            )
            for _ in range(len(original_column))
        ]
    else:
        return original_column


def fuzzy_input(input_file):
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
        for item in path_obj.parent.glob("./data/*2024*")
        if not_a_helper(item)  # exclude options files
    ]

    if input_options:
        input_file = input_options[0]
        print(f"Loading {input_file}")  # warn of change
        return input_file

    raise FileNotFoundError("Could not find input file in ./data/")


if __name__ == "__main__":
    # Load the original dataset

    original_csv_path = fuzzy_input(None)
    original_data = pd.read_csv(original_csv_path)

    # Randomize the rows of the orig dataset to keep freq of categorical data
    randomized_df = randomize_rows(original_data)

    # Generate the fake dataset
    fake_data = {}
    for column in original_data.columns:
        fake_data[column] = generate_fake_data(column, original_data[column])

    fake_df = pd.DataFrame(fake_data)
    fake_df = fake_df.iloc[:10]

    # Save the fake dataset to a new CSV
    fake_csv_path = "./data/example_input.csv"
    fake_df.to_csv(fake_csv_path, index=False)

    print(f"Fake dataset saved to {fake_csv_path}")
