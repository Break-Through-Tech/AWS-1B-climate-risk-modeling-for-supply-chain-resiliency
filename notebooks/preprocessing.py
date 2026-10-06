import pandas as pd
df = pd.load_csv('data.csv')
threshold = 0.5

#Establish SST climatology
sst_climatorgy = df.groupby(['site_id', 'month', 'year'])['sst'].mean().reset_index()
df = df.merge(sst_climatorgy, on=['site_id', 'month', 'year'], suffixes=('', '_climatology'))
df["ssta"] = df["sst"] - df["climatology_sst"]


#Create thresholds
def create_target(ssta):
    if ssta <= -threshold:
        return "cooling"
    elif ssta >= threshold:
        return "warming"
    else:
        return "normal"

#Create target classes
df["target_class"] = df["ssta"].apply(create_target)

#Counting values for imbalance
print(df["target_class"].value_counts())
print(df["target_class"].value_counts(normalize=True) * 100)