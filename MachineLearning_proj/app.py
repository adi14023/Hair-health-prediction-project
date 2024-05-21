import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

# load the models and encoders that were stored using pickle

model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb')) 


def main(): 
    st.title(" Hair health  Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Hair health  Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    # we need to create boxes for inputting all the independent features.

    age = st.text_input("Age","0") 

    gender = st.selectbox("Gender",["female","male"])

    job  = st.selectbox("job_role",['Government Employee', 'Employee', 'Jobless'])

    prov = st.selectbox("province",['Bengkulu', 'Bandung', 'Palu', 'Palangkaraya', 'Serang','Banda Aceh', 'Palembang', 'Kupang','Sofifi', 'Ambon','Tanjungselor', 'Tanjung Pinang', 'Banjarmasin', 'Denpasar','Mamuju', 'Makassar', 'Pangkalpinang', 'Yogyakarta','Pontianak','Mataram', 'Manokwari', 'Gorontalo', 'Semarang', 'Surabaya','Jakarta', 'Banda Lampung', 'Kendari', 'Pekanbaru','Jayapura','Jambi', 'Manado', 'Medan', 'Samarinda', 'Padang'])

    salary = st.text_input("salary","0")

    is_married = st.text_input("is_married","0")

    her = st.text_input("is_hereditary","0")

    weight = st.text_input("weight","0")

    height = st.text_input("height","0")

    shampoo = st.selectbox("shampoo",['Pantone', 'Moonsilk', 'Deadbuoy', 'Merpati', 'Shoulder & Head'])

    is_smoker = st.text_input("is_smoker","0")

    education = st.selectbox("education",['Bachelor Degree', 'Elementary School', 'Magister Degree',
    'Senior High School', 'Junior High School', 'Doctoral Degree'])

    stress = st.text_input("stress","0")


    
    if st.button("Predict"): 
        
        features = [[age,gender,job,prov,salary,is_married,her,weight,height,shampoo,is_smoker,education,stress]]
        data = {'age':float(age),'gender':gender,'job':job,'province':prov,'salary':float(salary),'is_married':float(is_married),'is_hereditary':float(her),'weight':float(weight),'height':float(height),'shampoo':shampoo,'is_smoker':float(is_smoker),'education':education,'stress':float(stress)}

        # create a dataframe with the inputs

        df=pd.DataFrame([list(data.values())], columns=['age','gender','job_role','province','salary','is_married','is_hereditary','weight','height','shampoo','is_smoker','education','stress'])

        # whatever preprocessing steps were done in preperation for training the model needs to be applied here.

        df['BMI'] = df['weight'] / (((df['height'])/100)**2)

        s = (df.dtypes == 'object')

        object_cols = list(s[s].index)

        new_data = pd.DataFrame(encoder.transform(df[object_cols]))

        new_data.index = df.index

        num_data = df.drop(object_cols, axis=1)


        # Add one-hot encoded columns to numerical features
        new_data = pd.concat([num_data, new_data], axis=1)

        # Ensure all columns have string type
        new_data.columns = new_data.columns.astype(str)

        # get the prediction
        
        pred = model.predict(new_data)

        if pred == 1:
            st.success('You have a higher chance or risk of baldness')
        else:
            st.success('You have a lower chance or risk of developing baldness')

      
if __name__=='__main__': 
    main()