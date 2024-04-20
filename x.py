import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import preprocessing
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

full_data = pd.read_csv('/home/dmesg/development/finalyearproject/BE-project-/Weather_data.csv')
# full_data.head()
full_data.isnull().sum()

# print("predicted Rainfall for one day ",y_pred)
# print("The Rainfall of the day in litre in 1 acre region ",water)
# #5.5 vakue is tempareture dependent 
# req=(5.5/1000)*(4046.86)*1000
# print("Required water in liters for one day in 1 acre ",req)
# print("The water in liters to be supplied will be ",(req-water))
# @app.get("/")
# async def tan():
#     return "Hello"



@app.get("/")
async def root(id:str ="tomato",hum:str="359",temp:str="2159",windspeed:str="81"): 
    full_data['Hum-min'].fillna(value=full_data['Hum-min'].mean(),inplace=True)
    full_data['Average Wind Speed'].fillna(value=full_data['Average Wind Speed'].mean(),inplace=True)
    full_data['Rainfall'].fillna(value=0.0,inplace=True)
    full_data.isnull().sum()
    oversampled=full_data
    oversampled.head()

    lencoders = {}
    for col in oversampled.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        oversampled[col] = lencoders[col].fit_transform(oversampled[col])
    warnings.filterwarnings("ignore")

    MiceImputed = oversampled.copy(deep=True) 
    mice_imputer = IterativeImputer()
    MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversampled)
    MiceImputed.head()
    MiceImputed.isnull().any()
    MiceImputed.shape

    corr = MiceImputed.corr()
#mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(20, 20))
    cmap = sns.diverging_palette(250, 25, as_cmap=True)
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})
    r_scaler = preprocessing.MinMaxScaler()
    r_scaler.fit(MiceImputed)
    modified_data = pd.DataFrame(r_scaler.transform(MiceImputed), index=MiceImputed.index, columns=MiceImputed.columns)
    modified_data.head()
    modified_data['Rainfall'].unique()
    data=MiceImputed
    
    X = data[['Max-Temp', 'Min-temp', 'Hum-max', 'Hum-min', 'Evaporation', 'Average Wind Speed']]  # Features
    y = data['Rainfall'] 


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model=XGBRegressor(objective ='reg:linear', n_estimators = 100, seed = 123)
    model.fit(X_train,y_train)
    # array first element is temp 
    #Third element id humidity
    #last element is wind speed
    print("hi",temp,hum,windspeed)
    y_pred=model.predict([[float(temp),19,float(hum),43,2.5,float(windspeed)]])
    
    # if summer or winter set the  water as 0 
    water=(y_pred[0]/1000)*(4046.86)*1000
    
  #  print("predicted Rainfall for one day ",y_pred)
  #  print("The Rainfall of the day in litre in 1 acre region ",water)
  #  RequiredWater = water[0]
#5.5 vakue is tempareture dependent 
    variableCal = {"tomato":8.2,"wheat":5.5,"onion":5.4,"maize":8.1,"soyabean":7.2}
    # water = 0
    
    req=(variableCal[id]/1000)*(4046.86)*1000
  #  print("Required water in liters for one day in 1 acre ",req)
 #   print("The water in liters to be supplied will be ",(req-water))
    print(req,req-water)
    return [req,water,req-water]
    #return {"Required water": req,"The rainfall of the day in litre in 1 acre region":[water]}
