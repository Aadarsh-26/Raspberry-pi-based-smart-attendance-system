import firebase_admin
from firebase_admin import credentials, storage,db
import numpy as np
import cv2
import pyttsx3
import csv
import os
from datetime import datetime
import keyboard
def time_cur(): #function for returning current time
    now=datetime.now()
    current_time=now.strftime("%H:%M:%S")
    return current_time
def get_data():
   # app2=firebase_admin.initialize_app(cred,{"databaseURL": "https://test-8e062-default-rtdb.firebaseio.com/"})
    ref = db.reference("/")
    return db.reference("/file/name").get()

def check_data(data,students,lnwrite):
    if data in students:
        print(data)
        name=known_names[known_id_no.index(data,0,5)]
        students.remove(data)
        lnwrite.writerow([name,time_cur(),known_roll[known_id_no.index(data,0,5)],"Present"])
        return f"Welcome {name}, have a good day "
    elif data in known_id_no or data=="0":
        return None
    else:
        return "No attandance"
def say_name(name):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(f"{name}")
    engine.runAndWait()
say_name("Good morning sir, Team IOT elites present your attandance assistant")
known_id_no=["200524","200580","200556","200558","200574"]
known_names=["Aadarsh","Hemant","Preksha","Prerna","Vansh"]
known_roll=["20001017001","20001017073","20001017043","20001017045","20001017066"]
students=known_id_no.copy()
cred = credentials.Certificate("./test-8e062-firebase-adminsdk-fiod6-81bd798af5.json")
app = firebase_admin.initialize_app(cred, { 'storageBucket' : 'test-8e062.appspot.com' ,"databaseURL": "https://test-8e062-default-rtdb.firebaseio.com/"})
# data=get_data()
now=datetime.now()
current_date=now.strftime("%Y-%m-%d") #variable current_date for finding the current date of running program
f=open(current_date+'.csv','w+',newline='') #opening of folder of .csv format and name according to current date
lnwrite= csv.writer(f) 
lnwrite.writerow(["Name of the student","Time of entry","Roll number","Status"])
while True:
    data=get_data()
    # print(data)
    name=check_data(data,students,lnwrite)
    if name!=None and name!="No attandance":
        say_name(name)
    if keyboard.is_pressed('esc'):
        break
    if len(students)==1:
        break



