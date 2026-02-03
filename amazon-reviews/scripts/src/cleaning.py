###########################################################
# Step 1 - Cleaning & Preprocessing (MULTICORE - JOBLIB)
# Author: @andvsilva
# Date: Sat Jan 31 2026
###########################################################

# ---------------------------------------------------------
# Libraries
# ---------------------------------------------------------
import time
from datetime import datetime
from multiprocessing import cpu_count
from contextlib import contextmanager
import pandas as pd
import sqlite3
from joblib import Parallel, delayed
from tqdm import tqdm
import toolkit as tool

# path ti the file sqlite
db_path = "../datasets/database.sqlite"

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def log(msg: str) -> None:
    print(f"[INFO] {msg}")

# ---------------------------------------------------------
# tqdm + joblib integration (OFFICIAL & SAFE)
# ---------------------------------------------------------
@contextmanager
def tqdm_joblib(tqdm_object):
    from joblib.parallel import BatchCompletionCallBack

    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = BatchCompletionCallBack
    try:
        import joblib.parallel
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    start_time = time.time()

    log("Cleaning the dataset...")
    log(f"Date: {datetime.now()}")

    # -----------------------------------------------------
    # Load dataset - connect to the database
    # -----------------------------------------------------
    conn = sqlite3.connect(db_path)

    # number of rows from the database
    n_rows = 500

    # lista as tabelas
    df_reviews = pd.read_sql(
        f"""
        SELECT Text, Score
        FROM Reviews
        WHERE Text IS NOT NULL
          AND Score IS NOT NULL
        LIMIT {n_rows}
        """,
        conn
    )

    total_rows = df_reviews.shape
    print(total_rows)
    print(df_reviews.head())

    conn.close()

    # -----------------------------------------------------
    # Text preprocessing (MULTICORE + PROGRESS BAR)
    # -----------------------------------------------------
    log("Applying text preprocessing (joblib + tqdm)...")

    texts = df_reviews["Text"].tolist()
    n_cores = max(cpu_count() - 1, 1)

    with tqdm_joblib(
        tqdm(total=len(texts), desc="Preprocessing Text")
    ):
        df_reviews["Text"] = Parallel(
            n_jobs=n_cores,
            backend="loky",
            batch_size="auto"
        )(
            delayed(tool.preprocess_text)(text)
            for text in texts
        )

    # -----------------------------------------------------
    # Save cleaned dataset
    # -----------------------------------------------------
    df_reviews.reset_index(drop=True, inplace=True)
    df_reviews.to_feather("../datasets/feather/cleaned.ftr")

    # -----------------------------------------------------
    # Finish
    # -----------------------------------------------------
    time_exec_min = round((time.time() - start_time) / 60, 4)

    log(f"Execution time: {time_exec_min} minutes")
    log("Cleaning step finished successfully.")
    log("Next step: Feature Engineering.")
    log("All Done.")

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
