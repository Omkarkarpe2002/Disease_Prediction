from tkinter import * 
import numpy as np
from joblib import dump, load

root = Tk()
root.title('Disease Prediction')
root.geometry('900x700')

symptoms = [
  'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]

heading = Label(root,text='Disease Prediction')
heading.config(font=('Times',30,'bold italic'))
heading.grid(row=1, column=0, columnspan=5,padx=200)

span = Label(root, text='')
span.grid(row=2,column=0, columnspan=2,pady=10)

s1 = Label(root, text='Symptom1')
s1.config(font=('Times',15))
s1.grid(row = 3, column=0)

s1_input = StringVar()
s1_input.set('None')
S1 = OptionMenu(root, s1_input, *symptoms)
S1.config(font=('Times',15))
S1.grid(row=3,column=1,pady=5)

s2 = Label(root, text='Symptom2')
s2.config(font=('Times',15))
s2.grid(row = 4, column=0)

s2_input = StringVar()
s2_input.set('None')
S2 = OptionMenu(root, s2_input, *symptoms)
S2.config(font=('Times',15))
S2.grid(row=4,column=1,pady=5)

s3 = Label(root, text='Symptom3')
s3.config(font=('Times',15))
s3.grid(row = 5, column=0)

s3_input = StringVar()
s3_input.set('None')
S3 = OptionMenu(root, s3_input, *symptoms)
S3.config(font=('Times',15))
S3.grid(row=5,column=1,pady=5)

s4 = Label(root, text='Symptom4')
s4.config(font=('Times',15))
s4.grid(row =6, column=0)

s4_input = StringVar()
s4_input.set('None')
S4 = OptionMenu(root, s4_input, *symptoms)
S4.config(font=('Times',15))
S4.grid(row=6,column=1,pady=5)

s5 = Label(root, text='Symptom5')
s5.config(font=('Times',15))
s5.grid(row =7, column=0)

s5_input = StringVar()
s5_input.set('None')
S5 = OptionMenu(root, s5_input, *symptoms)
S5.config(font=('Times',15))
S5.grid(row=7,column=1,pady=5)

s6 = Label(root, text='Symptom6')
s6.config(font=('Times',15))
s6.grid(row =3, column=3)

s6_input = StringVar()
s6_input.set('None')
S6 = OptionMenu(root, s6_input, *symptoms)
S6.config(font=('Times',15))
S6.grid(row=3,column=4,pady=5)

s7 = Label(root, text='Symptom7')
s7.config(font=('Times',15))
s7.grid(row =4, column=3)

s7_input = StringVar()
s7_input.set('None')
S7 = OptionMenu(root, s7_input, *symptoms)
S7.config(font=('Times',15))
S7.grid(row=4,column=4,pady=5)

s8 = Label(root, text='Symptom8')
s8.config(font=('Times',15))
s8.grid(row =5, column=3)

s8_input = StringVar()
s8_input.set('None')
S8 = OptionMenu(root, s8_input, *symptoms)
S8.config(font=('Times',15))
S8.grid(row=5,column=4,pady=5)

s9 = Label(root, text='Symptom9')
s9.config(font=('Times',15))
s9.grid(row =6, column=3)

s9_input = StringVar()
s9_input.set('None')
S9 = OptionMenu(root, s9_input, *symptoms)
S9.config(font=('Times',15))
S9.grid(row=6,column=4,pady=5)

s10 = Label(root, text='Symptom10')
s10.config(font=('Times',15))
s10.grid(row =7, column=3)

s10_input = StringVar()
s10_input.set('None')
S10 = OptionMenu(root, s10_input, *symptoms)
S10.config(font=('Times',15))
S10.grid(row=7,column=4,pady=5)

span2 = Label(root, text='')
span2.grid(row=8,column=0, columnspan=2)

label = Label(root, text='SVM: ')
label.config(font=('Times',20))
label.grid(row=9,column=0,padx=20,pady=20)

label = Label(root, text='RandomForest: ')
label.config(font=('Times',20))
label.grid(row=10,column=0,padx=20,pady=20)

label = Label(root, text='KNN: ')
label.config(font=('Times',20))
label.grid(row=11,column=0,padx=20,pady=20)



def make_predictions(model_name=None, test_data=None):
  classifier= load("./saved_models/" + str(model_name)+ ".joblib")
  if test_data is not None:
      result = classifier.predict(test_data)
      return result
  else:
    result = classifier.predict(test_features)
    test_accuracy = accuracy_score(test_labels, result)
    return test_accuracy

def show():
  symptom = []
  if(s1_input != 'None'):
    symptom.append(s1_input.get())
  if(s2_input != 'None'):
    symptom.append(s2_input.get())
  if(s3_input != 'None'):
    symptom.append(s3_input.get())
  if(s4_input != 'None'):
    symptom.append(s4_input.get())
  if(s5_input != 'None'):
    symptom.append(s5_input.get())
  if(s6_input != 'None'):
    symptom.append(s6_input.get())
  if(s7_input != 'None'):
    symptom.append(s7_input.get())
  if(s8_input != 'None'):
    symptom.append(s8_input.get())
  if(s9_input != 'None'):
    symptom.append(s9_input.get())
  if(s10_input != 'None'):
    symptom.append(s10_input.get())

  data = []
  for s in symptoms:
    if s in symptom:
      data.append(1)
    else:
      data.append(0)

  data = np.array(data)
  data = data.reshape(1,-1)

  print(data)
  
  label = Label(root, text=make_predictions('SVM',data)[0])
  label.config(font=('Times',15))
  label.grid(row=9,column=2)

  label = Label(root, text=make_predictions('RandomForest',data)[0])
  label.config(font=('Times',15))
  label.grid(row=10,column=2)

  label = Label(root, text=make_predictions('KNN',data)[0])
  label.config(font=('Times',15))
  label.grid(row=11,column=2)


button = Button(root, text='Predict',command=show)
button.config(font=('Times',20))
button.grid(row=12,column=0,columnspan=5,pady=20)

root.mainloop()