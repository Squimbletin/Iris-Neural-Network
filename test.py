import pandas as pd

#Testing importing the dataset

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
df = pd.read_csv(url)

print(df.shape)  # Should print (150, 5)
print(df.head())  # Should show first 5 rows

