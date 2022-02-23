# -*- coding: utf-8 -*-
"""chatbot.py


"""

import nltk
import random
import numpy as np
import json
import pickle
import pyttsx3
import datetime
import wikipedia
import pyowm
import pygame

import webbrowser
import speech_recognition as sr
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model

engine = pyttsx3.init()
voices = engine.getProperty('voices')
#
engine.setProperty('voice', voices[1].id)
volume = engine.getProperty('volume')
engine.setProperty('volume', 10.0)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 35)
lemmatizer = WordNetLemmatizer()

with open('intents.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


#

greetings = ['hey there', 'hello', 'hi', 'Hai', 'hey!', 'hey','hello lixa']
question = ['How are you?', 'How are you doing?']
# responses = ['I am good ', "I'm fine"]
var1 = ['who made you', 'who created you']
var2 = ['I_was_created_by_AbhishekJha_in_his_computer.', 'Abhishek', 'Some_guy_whom_i_never_got_to_know.']
var3 = ['Hospital near me', 'Doctor near me', 'Doctor', 'Hospital']
var4 = ['who are you', 'what is you name']
cmd1 = ['Pharmacy near me', 'Pharmacist', 'Medical shop near me']
cmd2 = ['Diabetes prevention tips', 'sugar prevention tips', 'Sugar']
cmd3 = ['what is Normal bp range', 'Blood pressure', 'High Bp', 'Low Bp']

cmd4 = ['Suggest some fever medicines', 'fever prevention','suggest some medicine for fever']
cmd5 = ['tell me some common cold symptoms', 'flu symptoms']
cmd6 = ['exit', 'close', 'goodbye', 'nothing','bye']
cmd7 = ['covid vaccine ki jaankaari','corona ka tika','corona ki vaccine ki jaankaari']
colrep = ['covid vaccine center', ' najdiki covid tika kendra', 'nearest covid vaccination center']
cmd8 = ['Book appointment', 'book doctor appointment']
cmd9 = ['emergency', 'ambulance']
repfr8 = ['thank you', 'thanks', 'dhanywaad']
repfr9 = ['youre welcome', 'glad i could help you']
bye = ["cya", "see you later", "goodbye", "have a good day", "bye", "cao", "see ya"]
bye_responses = ["have a nice day", "goodbye"]
age_patterns = ["how old", "how old are you?", "what is your age", "how old are you", "age?"]
age_responses = ["I get reborn after every compilation", "my owner is  23 years old !"]
name_patterns = ["what is your name", "what should i call you", "what's your name?", "who are you?",
                 "can you tell me your name"]
name_responses = ["you can call me Lixa!", "i am Lixa!", "i am Lixa your medical assistant"]

common_cold_symptoms = ["Runny or stuffy nose", "Sore throat",
                        "Cough",
                        "Congestion",
                        "Slight body aches or a mild headache",
                        "Sneezing",
                        "Low-grade fever",
                        "Generally feeling unwell (malaise)"]

common_cold_responses = "It seems that you are suffering from common cold"
fever_symptoms = ["Sweating",
                  "Chills and shivering",
                  "Headache",
                  "Muscle aches",
                  "Loss of appetite",
                  "Irritability",
                  "Dehydration",
                  "General weakness"],
fever_symptoms_responses = "It seems that you are suffering from fever"
Diabetes_symptoms_patterns = ["increased hunger",
                              "increased thirst",
                              "weight loss",
                              "frequent urination",
                              "blurry vision",
                              "extreme fatigue"]
Diabetes_symptoms_patterns_responses = "It seems that you are suffering from Diabetes"

Depression_symptoms_patterns = ["Hopeless outlook",
                                "Lost interest",
                                "Increased fatigue",
                                "sleep problem",
                                "Anxiety",
                                "change in weight",
                                "Looking at death",'stress'],
Depression_symptoms_patterns_responses = "It seem that you are suffering from depression Do not worry Treatment is " \
                                         "available "

Asthma_symptoms_patterns = ["coughing",
                            "tightness in the chest",
                            "shortness of breath",
                            "difficulty talking",
                            "panic",
                            "fatigue"]
Asthma_symptoms_patterns_responses = "It seem that you may be suffering from Asthma"

common_cold_medicines_patterns = ["What medicines can I buy to help me with my common cold?",
                                  "tell me some prevention method from common cold",
                                  "What should I eat or drink if i am suffering from common cold?",
                                  "How can I keep from getting a cold or the flu?"]
common_cold_medicines_patterns_responses = "medicines you can consume : Dextromethorphan,Decongestant," \
                                           "Diphenhydramine,Crocin Cold & Flu Max, preventions  that you must follow " \
                                           ":Wash your hands,Avoid touching your face,Clean frequently used surfaces " \
                                           "Use hand  sanitizers SUGGESTED FOODS ARE:Garlic,Vitamin C–containing " \
                                           "fruits,Leafy greens,Broccoli,Oatmeal,Spices Chicken Soup "

fever_medicine_patterns = ["What medicines can I buy to help me with my fever?",
                           "tell me some prevention method from fever",
                           "What should I eat or drink if i am suffering from fever?",
                           "How can I keep from getting a fever?", "fever medicines", "medicines for fever"],
fever_medicine_patterns_responses = "medicines you can consume : acetaminophen ibuprofen aspirin Crocin Cold & Flu " \
                                    "Max prevention that you must follow :Wash your hands,Cover your mouth when you " \
                                    "cough and your nose when you sneeze,Clean frequently used surfaces,Avoid sharing " \
                                    "cups, glasses, and eating utensils with other people.SUGGESTED FOODS ARE:Garlic " \
                                    "Vitamin C–containing fruits Leafy greens Broccoli,Oatmeal,Spices,Chicken Soup "

diabetes_medicine_patterns = ["What medicines can I buy to help me with my diabetes?",
                              "tell me some prevention method from diabetes",
                              "What should I eat or drink if i am suffering from diabetes?",
                              "How can I keep from getting diabetes?"]
diabetes_medicine_patterns_responses = [
    "medicines you can consume : Insulin ,Amylinomimetic drug,Dipeptidyl peptidase-4 (DPP-4) inhibitor, prevention "
    "that you must follow :Cut Sugar and Refined Carbs From Your Diet,Work Out Regularly,Drink Water as Your Primary "
    "Beverage,Lose Weight If You’re Overweight or Obese,Quit Smoking, Follow a Very-Low-Carb Diet,Watch Portion "
    "Sizes,SUGGESTED FOODS ARE:Leafy greens,Avocados,Eggs"]

depression_medicine_patterns = ["What medicines can I buy to help me with my depression?",
                                "tell me some prevention method from depression",
                                "What should I eat or drink if i am suffering from depression?",
                                "How can I keep from getting depression?"],
depression_medicine_patterns_responses = "medicines you can consume :  brexpiprazole, quetiapine,olanzapine, " \
                                         "prevention that you must follow :Exercise regularly,Cut back on social " \
                                         "media time,Drink Water as Your Primary Beverage,Build strong relationships " \
                                         "Minimize your daily choices, Follow a Very-Low-Carb Diet,SUGGESTED FOODS " \
                                         "ARE:Get Enough Vitamin D Include Omega-3 Fatty Acids,Beans and legumes "

asthma_medicine_patterns = ["What medicines can I buy to help me with my asthma?",
                            "tell me some prevention method from asthma",
                            "What should I eat or drink if i am suffering from asthma?",
                            "How can I keep from getting asthma?", "Medicines for asthma"]
asthma_medicine_patterns_response = [
    "medicines you can consume : epinephrine,anticholinergic,Proair HFA, prevention that you must follow : Identify "
    "Asthma Triggers, Stay Away From Allergens,Avoid Smoke of Any Type,SUGGESTED FOODS ARE:carrots,juice,eggs,"
    "broccoli,cantaloupe,milk"]

Consultation = ["who should i contact for consultation?", "is there any doctor available?",
                "can you give me some suggestions for doctor consultations?",
                "can you set up a meeting with a doctor for consultation?",
                "is there any doctor available for consultation"]
consultation_responses = [

    "https://www.1mg.com/online-doctor-consultation"
    "https://www.tatahealth.com/online-doctor-consultation/general-physician", " https://www.doconline.com/"
    ]

#
engine.say("Hello I am Lixa Your Health assistant How can I help you today")

engine.runAndWait()

while True:
    message = ""
    now = datetime.datetime.now()
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Tell me something:")
        audio = r.listen(source)

    try:
        message = r.recognize_google(audio)

    except sr.UnknownValueError:
        print("Could not understand audio")
        engine.say('I didnt get that. Say it again')

        engine.runAndWait()
    # message=input("")
    if (len(message) == 0):
        continue



    # if r.recognize_google(audio) in greetings:
    #     random_greeting = random.choice(greetings)
    #     print(random_greeting)
    #     engine.say(random_greeting)
    #     engine.runAndWait()
    #
    #
    # elif r.recognize_google(audio) in question:
    #     engine.say('I am fine')
    #     engine.runAndWait()
    #     print('I am fine')
    #
    # elif r.recognize_google(audio) in var1:
    #     reply = random.choice(var2)
    #     engine.say(reply)
    #     engine.runAndWait()
    #     print(reply)
    #
    # elif r.recognize_google(audio) in var3:
    #
    #     engine.say('Redirecting you to nearest hospital')
    #     webbrowser.open('www.google.com/hospital near me')
    #     engine.runAndWait()
    #
    #
    # elif r.recognize_google(audio) in var4:
    #
    #     engine.say("I am Lixa Your Health assistant")
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in cmd1:
    #
    #     engine.say('Finding pharmacy near you')
    #     webbrowser.open('www.google.com/pharmacy near me')
    #     engine.runAndWait()
    #
    #
    # elif r.recognize_google(audio) in cmd2:
    #
    #     engine.say('Lose extra weight Be more physically active  Eat healthy plant foods Eat healthy fats  Skip fad diets and make healthier choices')
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in cmd3:
    #     engine.say('Redirecting You to Blood pressure Chart ')
    #     webbrowser.open('https://www.idealbloodpressureinfo.com/wp-content/uploads/2013/09/blood-pressure-chart-by-age1.png')
    #     engine.runAndWait()
    #
    #
    #
    #
    # elif r.recognize_google(audio) in cmd4 or r.recognize_google(audio) in fever_medicine_patterns:
    #     engine.say(fever_medicine_patterns_responses)
    #     engine.runAndWait()
    #
    #
    # elif r.recognize_google(audio) in cmd5 or r.recognize_google(audio) in common_cold_medicines_patterns:
    #
    #     engine.say(common_cold_medicines_patterns_responses)
    #     engine.runAndWait()
    #
    #
    # elif r.recognize_google(audio) in cmd6 or r.recognize_google(audio) in bye:
    #     engine.say(random.choice(bye_responses))
    #     exit()
    #     engine.runAndWait()
    #
    #
    # elif r.recognize_google(audio) in cmd7:
    #     engine.say('Redirecting you to most frequently asked question')
    #     webbrowser.open('https://www.mohfw.gov.in/pdf/FAQsonCOVID19VaccineDecember2020.pdf')
    #     engine.runAndWait()
    #
    #
    # elif r.recognize_google(audio) in colrep:
    #     engine.say('Please wait')
    #     webbrowser.open('www.Google.com/nearest covid vaccination centre')
    #     engine.runAndWait()
    #
    #
    # elif r.recognize_google(audio) in cmd8 or r.recognize_google(audio) in Consultation:
    #
    #     engine.say("You can contact various doctors here for any kind of consultation:")
    #     webbrowser.open(random.choice(consultation_responses))
    #     engine.say("or you can pay a visit to your local area doctor or family doctor.")
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in repfr8:
    #     engine.say(random.choice(repfr9))
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in age_patterns:
    #     engine.say(random.choice(age_responses))
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in name_patterns:
    #     engine.say(random.choice(name_responses))
    #     engine.runAndWait()
    #
    #
    # elif r.recognize_google(audio) in common_cold_symptoms:
    #     engine.say(common_cold_responses)
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in fever_symptoms:
    #     engine.say(fever_symptoms_responses)
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in Diabetes_symptoms_patterns:
    #     engine.say(Depression_symptoms_patterns_responses)
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in Depression_symptoms_patterns:
    #     engine.say(Depression_symptoms_patterns_responses)
    #     webbrowser.open('https://www.google.com/search?q=depression+treatment')
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in Asthma_symptoms_patterns:
    #     engine.say(Asthma_symptoms_patterns_responses)
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in diabetes_medicine_patterns:
    #     engine.say(diabetes_medicine_patterns_responses)
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in depression_medicine_patterns:
    #     engine.say(depression_medicine_patterns_responses)
    #     engine.runAndWait()
    #
    # elif r.recognize_google(audio) in asthma_medicine_patterns:
    #     engine.say(asthma_medicine_patterns_response)
    #     engine.runAndWait()
    #
    #
    # else:
    #     engine.say("please wait")
    #     engine.runAndWait()
    #     webbrowser.open(r.recognize_google(audio))
    #
    #     engine.runAndWait()
    #     # userInput3 = input("or else search in google")
    #     # webbrowser.open_new('www.google.com/search?q=' + userInput3)





    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
    engine.say(res)
    engine.runAndWait()
