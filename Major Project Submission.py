#%%
#Major Prjoect Submission : IPL Score Prediction by : Milind Dalakoti @ milinddalakoti@gmail.com
#%%
import pandas as pd 
df=pd.read_csv("ipl2017.csv")
#%%
df.head()
#Checking for null values
df.isna().sum() 

# %%
y=df["total"]
#dropping total [Target column ] and unwnated columns
x=df.drop(["total","bowl_team","bat_team","date"],axis=1)

x.head()

# %%
# #finding Number of unique values in each column
len(x["venue"].unique())
#%%
len(x["batsman"].unique().sum())
#%%
len(x["bowler"].unique().sum())

# %%
type(x)
#%%
from sklearn.preprocessing import  LabelEncoder
encoder=LabelEncoder()

#%%
encoder.fit(x["venue"])
x["venue"]=encoder.transform(x["venue"])
encoder.fit(x["batsman"])
x["batsman"]=encoder.transform(x["batsman"])
encoder.fit(x["bowler"])
x["bowler"]=encoder.transform(x["bowler"])

#%%

# %% 
#Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=26)
#%%
#Scaling Data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
#Fitting and Transforming data
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# %%
#Using Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)


# %%
#Finding Accuracy
model.score(x_test,y_test)*100

#%% Prediction 
data = pd.DataFrame({"mid":[1],"venue":[14],"batsman":[328],"bowler":[96],"runs":[38],"wickets":[0],"overs":[1.2],"runs_last_5":[10],"wickets_last_5":[0],"striker":[12],"non-striker":[2]})
data=scaler.transform(data)
model.predict(data)
# %%
