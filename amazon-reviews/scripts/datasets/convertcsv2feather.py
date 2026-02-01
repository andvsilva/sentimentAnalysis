import pandas as pd


print("saving the file format feather...")

# Load CSV
df = pd.read_csv("Reviews.csv")

df = df.reset_index(drop=True)

# Save as Feather
df.to_feather("feather/Reviews.ftr")