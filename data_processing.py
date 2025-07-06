import pandas as pd

df_out = pd.DataFrame(columns=['text', 'is-sexist'])


####### About the gesis dataset

df_gesis = pd.read_csv('gesis/sexism_data.csv', sep=',', encoding='utf-8', usecols=['text', 'sexist'])

df_gesis['sexist'] = df_gesis['sexist'].astype(int)

df_gesis = df_gesis.rename(columns={'sexist': 'is-sexist'})

df_out = pd.concat([df_out, df_gesis], ignore_index=True)

####### About the online-misogyny dataset

df_eacl = pd.read_csv('online-misogyny-eacl2021-main/data/final_labels.csv', sep=',', encoding='utf-8', usecols=['body', 'level_1'])

df_eacl['level_1'] = df_eacl['level_1'].apply(lambda x: 1 if x=='Misogynistic' else 0)

df_eacl = df_eacl.rename(columns={'body': 'text'})
df_eacl = df_eacl.rename(columns={'level_1': 'is-sexist'})

df_out = pd.concat([df_out, df_eacl], ignore_index=True)

####### Create the .csv

df_out.to_csv("dataset.csv", index=False)