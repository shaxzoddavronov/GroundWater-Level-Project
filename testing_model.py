import tensorflow as tf
import sklearn
import pandas as pd
import joblib
import pickle


model=tf.keras.models.load_model('lstm_sequential_model.keras')
pipeline=joblib.load('pipeline.pkl')

sample_data_featues={
    'GWL (m)':[0.82,0.96,1.06,1.05,1.02,0.99],
    'Well number':[1722155,1722155,1722155,1722155,1722155,1722155],
    'Rainfall mm':[0.149526, 0.058863, 0.129229, 0.010825, 0.000000, 0.000000],
    'Min air temp. C':[0.25766871, 0.30265849, 0.41308793, 0.5398773 , 0.58486708, 0.719836],
    'Max air temp. C':[0.41685144, 0.39689579, 0.68070953, 0.76718404, 0.8713969, 0.929047],
    'Ave. air temp. C':[0.2835443 , 0.30379747, 0.45063291, 0.71898734, 0.83797468, 0.881013]}

sample_df=pd.DataFrame(sample_data_featues)

cat_cols=['Well number']
numeric_cols=['Rainfall mm', 'Min air temp. C', 'Max air temp. C', 'Ave. air temp. C']

df=sample_df.copy()
df[cat_cols+numeric_cols]=pipeline.transform(df[cat_cols+numeric_cols])
result=model.predict(df[['GWL (m)']+numeric_cols].head(5).values.reshape(1,5,5))
print(f"Result >>>>>>>>>>> {result[0][0]}")