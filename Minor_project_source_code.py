#%%
import pandas as pd 
df=pd.read_csv("heart.csv")
#%%
df.head()

# %%
y=df["sex"]
# %%
x=df.drop("sex",axis=1)
x.head()
# %%
x.isna().sum()
# %%
x.dtypes

# %%
type(x)
# %%
x.shape
# %%
