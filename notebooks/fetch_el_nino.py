from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
el_nino = fetch_ucirepo(id=122) 
  
# data (as pandas dataframes) 
df = el_nino.data.features.copy()

# # metadata 
# print(el_nino.metadata) 

# # variable information 
# print(el_nino.variables) 

# creating a new 4-digit year column
df["full_year_val"] = df["year"] + 1900

# Create datetime using the new year column
df["datetime"] = pd.to_datetime(
    {
        "year": df["full_year_val"],
        "month": df["month"],
        "day": df["day"]
    }
)

# print(df.head(6)) # testing --> works!
