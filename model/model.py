import pickle
import numpy as np 
import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

# Creating the GUI
st.set_page_config(page_title = "Symptoms Checker", page_icon = ":male-doctor:", layout = "centered")
st.title("Symptoms Checker and Recommendation System")
st.info("This is a machine learning model which predicts potential disease based on the inputted symptoms and provides personalized recommendations for effective precautions, medicines, workout and diet to support proactive healthcare and wellness.")

# importing datasets  
dataset = pd.read_csv("datasets/Training.csv/Training.csv")
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# Assigning numerical value to each symptom
symptoms_dict = {}
keys_const = np.arange(0,132)
values_const = []
for col in dataset.columns:
    values_const.append(col)
for i in range(len(keys_const)):
    symptoms_dict.update({values_const[i]: keys_const[i]}) 

# Data Preprocessing
X = dataset.drop("prognosis",axis=1)
y = dataset["prognosis"]
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)
value_disease = np.unique(y)
key_disease = np.arange(0,41)
diseases_list={}
for j in range(len(key_disease)):
    diseases_list.update({key_disease[j]: value_disease[j]})
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=20)

# Helper Function
def helper(dis):
    desc = description[description["Disease"]==predicted_disease]["Description"]
    desc = "Description : " + "  ".join([w for w in desc])
    pre = precautions[precautions["Disease"]==dis][["Precaution_1","Precaution_2","Precaution_3","Precaution_4"]]
    pre = [col for col in pre.values]
    med = medications[medications["Disease"]==dis]["Medication"]
    med = [med for med in med.values]
    die = diets[diets["Disease"]==dis]["Diet"]
    die = [die for die in die.values]
    wrkt = workout[workout["disease"]==dis]["workout"]
    return desc,pre,med,die,wrkt

# Training the model
models={
    "SVC":SVC(kernel="linear"),
    "RandomForest":RandomForestClassifier(n_estimators=100,random_state=42),
    "GradientBooster":GradientBoostingClassifier(n_estimators=100,random_state=42),
    "KNeighbors":KNeighborsClassifier(n_neighbors=5),
    "MultinomialNB": MultinomialNB()
}

for model_name, model in models.items():
    #train model
    model.fit(X_train,y_train)
    #test model
    predictions = model.predict(X_test)
    #test accuracy
    accuracy = accuracy_score(y_test,predictions)
    #calculate confusion matrix
    cm = confusion_matrix(y_test,predictions)

svc = SVC(kernel="linear")
svc.fit(X_train,y_train)
ypred = svc.predict(X_test)
accuracy_score(y_test,ypred)

pickle.dump(svc,open("datasets/svc.pkl",'wb'))
svc=pickle.load(open("datasets/svc.pkl",'rb'))

# model prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]]=1
    return diseases_list[svc.predict([input_vector])[0]]

# Taking input symptoms
symptoms = st.multiselect("Select or Search your Symptoms",['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 
'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 
'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']) 

# Buttons and Result
with st.container():
    disease, recommendations = st.columns([1,3]) 

    predicted_disease = get_predicted_value(symptoms)
    desc,pre,med,die,wrkt = helper(predicted_disease)

    if st.button("Predict Disease and Show Recommendations"):
        with disease:
            if len(symptoms) > 0:
                st.write("Predicted Disease is: ")
                st.error(predicted_disease)
                st.info(desc)
               
                st.success("Suggested Medications:")
                for m_i in med:
                    m_i = m_i.lstrip("[")
                    m_i = m_i.rstrip("]")
                    m_real = m_i.split(",")
                    i=1
                    for item in m_real:
                        st.write(i,":",item)
                        i+=1

            else:
                st.error("No Symptoms are Selected!!")

        with recommendations:
            if len(symptoms) > 0:
                st.info("Suggested Precautions:")
                i=1
                for pc_i in pre[0]:
                    st.write(i,":",pc_i)
                    i+=1
                st.info("Suggested Workouts:")
                i=1
                for w_i in wrkt:
                    st.write(i,":",w_i)
                    i+=1
                st.success("Suggested Diet:")
                for d_i in die:
                    d_i = d_i.lstrip("[")
                    d_i = d_i.rstrip("]")
                    st.write(d_i)

            else:
                st.error("No Symptoms are Selected!!")

    # Reset button
    if st.button("Reset"):
        symptoms = []

    # Signature
    st.markdown("Built with hope ~aas:heart:")


