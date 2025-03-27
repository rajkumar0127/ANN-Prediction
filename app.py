import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import numpy as np
import pandas as pd
import pickle

# Load models
model=tf.keras.models.load_model('model.h5')

#Load encoder and scaler
with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender=pickle.load(f)
with open('oneHotEncoder_geo.pkl','rb') as f:
   oneHotEncoder_geo=pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)


st.title('Customer churn prediction')

geography=st.selectbox('Geography', oneHotEncoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,99)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_cad=st.selectbox('Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_cad],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})



geo_encoded=oneHotEncoder_geo.transform([[geography]])
geo_df=pd.DataFrame(geo_encoded,columns=oneHotEncoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)

input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

if prediction >.5:
    st.write('The coustomer is likely to churn')
    st.write('churn probability: ',prediction)
else:
    st.write('The coustomer is not likely to churn')
    st.write('churn probability: ',prediction)