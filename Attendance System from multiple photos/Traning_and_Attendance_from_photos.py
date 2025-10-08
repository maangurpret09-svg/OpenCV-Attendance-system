import cv2 as cv

import numpy as np

import json

import csv

from datetime import datetime

import os

import glob

from PIL import Image

from pillow_heif import register_heif_opener

register_heif_opener()

import pandas as pd

import winsound

#Paths

face_recognizer_model_file = '' # Use your trained model file here

mapping = "label_mapping.json"

attendance_file = "attendance.csv"

haar_file = "haar_face.xml"

photo_folder = "photos"

haar_cascade = cv.CascadeClassifier(haar_file)
if haar_cascade.empty():
    print("Haar cascade xml file not found or failed to load.") 
    exit()
    
Students = [] #Empty Student list
face_recognizer = cv.face.LBPHFaceRecognizer_create()

if not os.path.exists(face_recognizer_model_file) or not os.path.exists(mapping):
    print("Training Model...")
    
    face_samples = []
    labels = []
    
    capture = cv.VideoCapture(0)
    
    for label , (roll_number , name) in enumerate(Students):
        print(f"Collecting samples for {name} ({roll_number}). Press 'q' to stop early.")
        count = 0
        
        while True:
            istrue , frames = capture.read()
            if not istrue:
                print("Failed to capture image")
                break
            
            gray = cv.cvtColor(frames , cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray , scaleFactor=1.1 , minNeighbors=5)
            
            for (x , y , w , h) in faces_rect:
                faces_roi = gray[y : y + h , x : x + w]
                faces_roi = cv.resize(faces_roi, (200, 200))  # Resize to a standard size
                
                face_samples.append(faces_roi)
                labels.append(label)
                count += 1
                
                #Showing live preview with display box
                
                cv.putText(frames , f"{name} ({roll_number}) ({count})" , (x , y - 10) , cv.FONT_HERSHEY_SIMPLEX , 0.9 , (255 , 255 , 255) , 2)
                cv.rectangle(frames , (x , y ) , (x + w , y + h), (0 , 255 , 0) , 2)
                
            cv.imshow("Training Samples" , frames)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            if count >= 30:
                break

    capture.release()
    cv.destroyAllWindows()
    
    face_samples = np.array(face_samples , dtype='uint8')

    face_recognizer.train(face_samples , np.array(labels))
    face_recognizer.save(face_recognizer_model_file)

    print("Training complete and model saved.")

    label_mapping = {i : {roll_number : name} for i , (roll_number , name) in enumerate(Students)}
    with open(mapping , "w") as f:
        json.dump(label_mapping , f , indent = 4)
    print("Label mapping saved")

else:
    print("Model already exits and loading...")
    face_recognizer.read(face_recognizer_model_file)
    with open(mapping , "r") as f:
        label_mapping = json.load(f)
        
#-------------------------Photo Conversion---------------------------------
converted_folder = "photos_converted"

os.makedirs(converted_folder , exist_ok=True) # Create directory if it doesn't exist
        
#converting HEIC images to JPG
def convert_heic_in_folder(folder):
    converted_photos = []

    for heic_file in glob.glob(os.path.join(folder , "*.heic")):
        try:
            img = Image.open(heic_file)
            jpg_file = os.path.join(converted_folder, os.path.basename(heic_file).replace(".heic", ".jpg"))
            if not os.path.exists(jpg_file):
                img.save(jpg_file , format="JPEG" , quality=90)
                converted_photos.append(jpg_file)
                print(f"Converted {heic_file} to {jpg_file}")
            else:
                print(f"{jpg_file} already exists. Skipping conversion.")
        except Exception as e:
            print(f"Error converting {heic_file}: {e}")
            
    converted_photos.extend(glob.glob(os.path.join(folder , "*.jpg")))
    return converted_photos
    
#-------------------------Attendance Function---------------------------------
marked_today = set() # To keep track of students whose attendance has been marked today
def mark_attendance(name , roll_number , timestamp = None):
    if roll_number in marked_today:
        return # Attendance already marked for this session
    
    today = datetime.now().strftime("%d-%m-%y")
    time_now = timestamp if timestamp else datetime.now().strftime("%H:%M:%S")
    
    file_name = f"attendance_{today}.csv" 
    
    #Check if file exists or not
    is_file = os.path.exists(file_name)
    
    #Opening in append mode , if files doesn't exist it will create the file
    with open(file_name , "a+" , newline="") as f:
        writer = csv.writer(f) 
        if not is_file:
            writer.writerow(["Name" , "Roll_Number", "Date" , "Time"])
            
        writer.writerow([name , roll_number , today , time_now])
        print(f"Attendance Marked for {name} {roll_number} at {time_now} on {today}")
        
        marked_today.add(roll_number)
    try:        
        winsound.Beep(1000 , 200) # Beep sound for successful attendance marking
    except ImportError:
        print("\a")  # Fallback beep for non-Windows systems
        
#-------------------------Monthly Attendance---------------------------------
def monthly_attendance():
    month = datetime.now().strftime("%m-%y")
    Excel_file_name = f"Attendance_{month}.xlsx"
    
    #Collecting Daily attendance files of that month
    daily_files = [f for f in os.listdir() if f.startswith("attendance_") and f.endswith(".csv") and month in f]
    
    if not  daily_files:
        print("No attendance files found for this month.")
        return

    df_file = [pd.read_csv(file) for file in daily_files]
    final_file = pd.concat(df_file , ignore_index = True)
    
    #Save as a Excel file
    final_file.to_excel(Excel_file_name , index = False)
    print(f"Monthly attendance saved under the name {Excel_file_name}")
    
#-------------------------Image Processing ---------------------------------
#Processing each photo for attendance
def process_photo():
    for photo in converted_photo: 
        print(f"Processing {photo} for attendance...")
        image = cv.imread(photo)
        if image is None:
            print(f"Failed to load image {photo}. Skipping.")
            continue    
        gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray , scaleFactor=1.1 , minNeighbors=5)
        
        if len(faces_rect) == 0:
            print(f"No face detected in {photo}. Skipping.")
            continue
        for (x , y , w , h) in faces_rect:
            faces_roi = gray[y : y + h , x : x + w]
            faces_roi = cv.resize(faces_roi , (200 , 200))
            
            label , confidence = face_recognizer.predict(faces_roi)
            
            if confidence < 75: #Confidence threshold
                roll_number , name = list(label_mapping[str(label)].items())[0]
                timestamp = datetime.now().strftime("%H-%M-%S")
                mark_attendance(name , roll_number , timestamp)
                
                cv.putText(image , f"{name} ({roll_number})" , (x , y - 10) , cv.FONT_HERSHEY_SIMPLEX , 0.9 , (255 , 255 , 255) , 2)
                cv.rectangle(image , (x , y ) , (x + w , y + h), (0 , 255 , 0) , 2)
            else:
                cv.putText(image , "Unknown" , (x , y - 10) , cv.FONT_HERSHEY_SIMPLEX , 0.9 , (255 , 255 , 255) , 2)
                cv.rectangle(image , (x , y ) , (x + w , y + h), (0 , 0 , 255) , 2)
                print(f"Face not recognized in {photo}.")   
            
        cv.imshow("Recognition" , image)
        cv.waitKey(500) # Display each image for 0.5 seconds
        cv.destroyWindow("Recognition")
    
converted_photo = convert_heic_in_folder(photo_folder)
    
process_photo()

monthly_attendance()


print(f"Total students marked today: {len(marked_today)}")
