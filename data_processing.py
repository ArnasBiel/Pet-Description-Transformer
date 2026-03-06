# Data processing code
# Reads raw petfinder CSVs and serialises each entry into a pickle file

import polars as pl
import pickle

FILES = ["./petfinder_study1.csv", "./petfinder_study2.csv"]
OUTPUT_FILE = "./pet_description_dataset.pkl"


def create_entry(species, color, age, gender, size, text):
    """Build a structured text entry from pet attributes.
    Unfortunately no name information — could use a model to find names from descriptions?
    """
    entry = (
        f"Species: {species}\n"
        f"Color: {color}\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n"
        f"Size: {size}\n"
        f"Description:\n"
        f"<START_DESCRIPTION>{text}<END_DESCRIPTION>"
    )
    return entry


def process_files():
    for file in FILES:
        df = pl.read_csv(file, infer_schema_length=10000)

        with open(OUTPUT_FILE, "ab") as fp:
            for row in df.iter_rows():
                entry = create_entry(
                    species=row[4],
                    color=row[3],
                    age=row[6],
                    gender=row[7],
                    size=row[8],
                    text=row[1],
                )
                pickle.dump(entry, fp)  # we can read this out like a list with pickle
                print(entry)


if __name__ == "__main__":
    process_files()
