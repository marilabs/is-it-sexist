import pandas as pd

# Load dataset
df = pd.read_csv('dataset.csv', sep=',')

# Total counts overall
total_sexist = len(df[df['is-sexist'] == 1])
total_non_sexist = len(df[df['is-sexist'] == 0])

# Comments mentioning 'women' (case-insensitive)
contains_women = df[df['text'].str.contains("women", case=False, na=False)]

# Sexist comments mentioning 'women'
sexist_with_women = contains_women[contains_women['is-sexist'] == 1]
non_sexist_with_women = contains_women[contains_women['is-sexist'] == 0]

# Compute percentages
percent_sexist_with_women = 100 * len(sexist_with_women) / total_sexist if total_sexist > 0 else 0
percent_non_sexist_with_women = 100 * len(non_sexist_with_women) / total_non_sexist if total_non_sexist > 0 else 0

print(f"Total sexist comments: {total_sexist}")
print(f"Total non-sexist comments: {total_non_sexist}")
print(f"Sexist comments mentioning 'women': {len(sexist_with_women)} ({percent_sexist_with_women:.2f}%)")
print(f"Non-sexist comments mentioning 'women': {len(non_sexist_with_women)} ({percent_non_sexist_with_women:.2f}%)")