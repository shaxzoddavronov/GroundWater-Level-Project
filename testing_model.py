import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
import joblib
import pickle


model=tf.keras.models.load_model('lstm_sequential_model.keras')
pipeline=joblib.load('pipeline.pkl')

actual_values=[0.67, 0.83, 0.94,]
df=pd.read_excel('input_data_gwl.xlsx')

object_cols=df.select_dtypes(include='object').columns.tolist()[1:]
def convert_str_to_float(columns=object_cols):
  for col in columns:
    df[col]=df[col].apply(lambda x: x.replace(' ','').replace(',','.') if isinstance(x,str) else x).astype(np.float64)
  return df

df=convert_str_to_float()

cat_cols=['Well number']
numeric_cols=['Rainfall mm', 'Min air temp. C', 'Max air temp. C', 'Ave. air temp. C']
df[cat_cols+numeric_cols]=pipeline.transform(df[cat_cols+numeric_cols])

result=model.predict(df[['GWL (m)']+numeric_cols].head(12).values.reshape(1,12,5))[0]

for idx,(pred,actual) in enumerate(zip(result,actual_values),start=1):
    print(f"Prediction of {idx}-month: {pred}")
    print(f"Actual  of {idx}-month: {actual}")
