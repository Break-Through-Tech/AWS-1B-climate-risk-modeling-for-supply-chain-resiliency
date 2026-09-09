from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
el_nino = fetch_ucirepo(id=122) 
  
# data (as pandas dataframes) 
X = el_nino.data.features 
y = el_nino.data.targets 
  
# metadata 
print(el_nino.metadata) 
  
# variable information 
print(el_nino.variables) 
