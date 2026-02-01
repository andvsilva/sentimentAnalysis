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
from joblib import Parallel, delayed
from tqdm import tqdm

import toolkit as tool

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
    # Load dataset
    # -----------------------------------------------------
    df_reviews = pd.read_feather("../datasets/feather/Reviews.ftr")
    log(f"Dataset loaded | shape: {df_reviews.shape}")

    # Reduce memory usage
    df_reviews = tool.reduce_mem_usage(df_reviews)

    # -----------------------------------------------------
    # OPTIONAL: sample (REMOVE IN FINAL VERSION)
    # -----------------------------------------------------
    df_reviews = df_reviews.sample(200_000, random_state=42)

    # -----------------------------------------------------
    # Drop unused columns
    # -----------------------------------------------------
    rm_cols = [
        "Id",
        "ProductId",
        "UserId",
        "ProfileName",
        "HelpfulnessNumerator",
        "HelpfulnessDenominator",
        "Time",
        "Summary",
    ]

    df_reviews.drop(columns=rm_cols, inplace=True)

    # -----------------------------------------------------
    # Basic filtering
    # -----------------------------------------------------
    df_reviews.dropna(inplace=True)
    df_reviews = df_reviews[df_reviews["Score"] != 3]

    log(f"After filtering | shape: {df_reviews.shape}")

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
