import pandas as pd
import numpy as np
import datetime
import pickle
import lightgbm as lgb
import streamlit as st

def main():
    html_temp="""
    <div style = "background-color: Lightblue;padding:16px">
    <h2 style="color:black;text-align:center;"> Car Price Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    st.write(" ")
    st.write(" ")    
        
    model = pickle.load(open("model.pkl", "rb"))
        
    with open('encoding_mapping.txt', 'r') as file:
        lines = file.readlines()

    encoding_mapping_list = []

    for line in lines:
        line = line.strip()

        if line:
            encoding_mapping = eval(line)
            encoding_mapping_list.append(encoding_mapping)

    encoding_mapping = encoding_mapping_list[0] 
    
    s1 = st.selectbox("What is the company name of the car?",('Citroën', 'Nissan', 'Opel', 'Audi', 'Ford', 'Hyundai', 'Škoda', 'Volkswagen', 'Fiat', 'Renault', 'Jeep', 'Peugeot', 'Mercedes-Benz', 'Mitsubishi', 'BMW', 'Porsche', 'Toyota', 'Suzuki', 'Volvo', 'Mini', 'Seat', 'Dodge', 'Dacia', 'Honda', 'Mazda', 'Kia', 'Subaru', 'Lexus', 'Saab', 'Smart', 'Alfa Romeo', 'Chevrolet', 'Inny', 'Land Rover', 'Jaguar', 'Infiniti', 'SsangYong', 'Wartburg', 'MG', 'Aston Martin', 'Chrysler', 'Daewoo', 'Lancia', 'Triumph', 'Maserati', 'Daihatsu', 'Acura', 'Ferrari', 'Uaz', 'Aixam', 'Lincoln', 'Bentley', 'Cadillac', 'Polonez', 'Lada', 'Rover', 'Tesla', 'Microcar', 'Ligier', 'Pontiac', 'Isuzu', 'GMC', 'Austin', 'Rolls-Royce', 'Brilliance', 'Scion', 'Żuk', 'Lamborghini', 'Buick', 'Hummer', 'Trabant', 'Grecav', 'Morgan', 'LTI', 'Iveco', 'Tata', 'Tavria', 'McLaren', 'Moskwicz', 'Oldsmobile', 'Mercury', 'Gaz', 'Plymouth', 'Mahindra', 'Tarpan', 'Chatenet', 'Zaporożec' 'Yugo', 'Maybach', 'Gonow'))
    
    if s1 in encoding_mapping['Company']:
        p1 = encoding_mapping['Company'][s1]
    else:
        print("Please select company name.")
            
    s2 = st.selectbox("What is the body car type?",('Kompakt/Hatchback', 'SUV', 'Coupe', 'Sedan', 'Auta miejskie/małe', 'Minivan/Kombi', 'Kabriolet', 'Missing'))
    
    if s2 in encoding_mapping['Typ']:
        p2 = encoding_mapping['Typ'][s2]
    else:
        print("Please select type name.")
        
    s3 = st.selectbox("What is the car location?",('warminsko-mazurskie', 'mazowieckie', 'wielkopolskie', 'opolskie', 'podkarpackie', 'pomorskie', 'slaskie', 'lodzkie', 'dolnoslaskie', 'zachodniopomorskie', 'lubelskie', 'podlaskie', 'swietokrzyskie', 'kujawsko-pomorskie', 'malopolskie', 'lubuskie'))
    
    if s3 in encoding_mapping['Region']:
        p3 = encoding_mapping['Region'][s3]
    else:
        print("Please select region name.")

    s4 = st.selectbox("What is the color of the car ?",('Czarny', 'Bordowy', 'Srebrny', 'Zielony', 'Inny kolor', 'Szary', 'Biały', 'Czerwony', 'Niebieski', 'Beżowy', 'Fioletowy', 'Brązowy', 'Złoty', 'Żółty')) 
    
    if s4 in encoding_mapping['Color']:
        p4 = encoding_mapping['Color'][s4]
    else:
        print("Please select region name.")    
        
    s5 = st.selectbox("Is the transmission front wheel drive?", ('Tak', "Nie"))

    if s1 == 'Tak':
        p5 = 1
    else:
        p5 = 0   
    
    s5 = st.selectbox("Is the car damaged?", ('Tak', 'Nie'))

    if s1 == 'Tak':
        p6 = 0
    else:
        p6 = 1
        
    p7 = st.number_input("What is the engine capacity (in cm3)?", step = 1)
    
    p8 = st.number_input("What is the engine power (in hp)?", step = 1)
    
    p9 = st.number_input("What is the millage (in thousend km)?", step = 1)
    
    date_time = datetime.datetime.now()
    
    year = st.number_input("What is the year of production of the car?", step = 1)
    
    p10 = date_time.year - year
    
    data_new = pd.DataFrame({
        'Norm Engine capacity': p7,
        'Norm Mileage': p8,        
        'Norm Engine power': p9,
        'Norm Age': p10,
        'norm_Typ': p2,
        'norm_Color': p4,
        'norm_Region': p3,
        'norm_Company': p1,        
        'Transmission_Na przednie koła': p5,
        'Damaged_Nie': p6
    }, index=[0])
    
    log_col = ['Norm Engine capacity', 'Norm Engine power', 'Norm Age']
    
    data_new[log_col] = np.log(data_new[log_col])
    
    col_standard_scaler = ['Norm Engine capacity', 'Norm Mileage', 'Norm Engine power', 'Norm Age']
    
    max_clip_value = 1e308
    
    data_new[col_standard_scaler] = np.clip(data_new[col_standard_scaler], -max_clip_value, max_clip_value)
    
    with open('scaler.pkl','rb') as f:
        sc = pickle.load(f)
        
    data_new[col_standard_scaler] = sc.transform(data_new[col_standard_scaler])
    
    try:
        if st.button('Predict'):
            pred = model.predict(data_new)
            pred = np.exp(pred)
            if pred > 0:
                st.success("You can sell your car for {} PLN.".format(pred))
            else:
                st.warning("You can not able to sell this car!")
    except:
        st.warning("Something Went Wrong! Please try again!")
    
if __name__ == '__main__':
    main()