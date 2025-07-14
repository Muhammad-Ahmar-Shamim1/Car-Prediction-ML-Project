import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

 
df = pd.read_csv(r"E:\Projects\deep_learning\project\used_cars.csv")
  
df['price'] = df['price'].replace('[\\$,]', '', regex=True).astype(float)
df['milage'] = df['milage'].str.replace(' mi.', '').str.replace(',', '')
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')

 
df = df[['model_year', 'milage', 'price']].dropna()
 
X = df[['model_year', 'milage']]
y = df['price']

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

 
pickle.dump(model, open("car_model.pkl", "wb"))
print("✅ Model saved as car_model.pkl")
