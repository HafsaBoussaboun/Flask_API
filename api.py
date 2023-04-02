import io
import json
import numpy as np
from PIL import Image
from flask import Flask, make_response, request, jsonify
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained machine learning model
age_model = load_model('age_model_pretrained.h5')
emotion_model = load_model('emotion_model_pretrained.h5')
gender_model = load_model('gender_model_pretrained.h5')
currency_model=load_model('model_currency.h5')

#define classes
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
emotion_ranges= ['happy','sad','neutral']


@app.route('/predictPersons', methods=['POST'])
def predictPersons():
    image_bytes = request.get_data()

   # Convert the image data to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image using cv2
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Perform some processing on the image
    test_image = img.copy()
    gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    i = 0
    for (x,y,w,h) in faces:
        i = i+1
        cv2.rectangle(test_image,(x,y),(x+w,y+h),(203,12,255),2)

        img_gray=gray[y:y+h,x:x+w]

        emotion_img = cv2.resize(img_gray, (48, 48), interpolation = cv2.INTER_AREA)
        emotion_image_array = np.array(emotion_img)
        emotion_input = np.expand_dims(emotion_image_array, axis=0)
        output_emotion= emotion_ranges[np.argmax(emotion_model.predict(emotion_input))]
        
        gender_img = cv2.resize(img_gray, (100, 100), interpolation = cv2.INTER_AREA)
        gender_image_array = np.array(gender_img)
        gender_input = np.expand_dims(gender_image_array, axis=0)
        output_gender=gender_ranges[np.argmax(gender_model.predict(gender_input))]

        age_image=cv2.resize(img_gray, (200, 200), interpolation = cv2.INTER_AREA)
        age_input = age_image.reshape(-1, 200, 200, 1)
        output_age = age_ranges[np.argmax(age_model.predict(age_input))]
    
        response = {
        'emotion': str(output_emotion),
        'age':str(output_age),
        'gender':str(output_gender)
    }
    return jsonify(response)



#######################################################

@app.route('/predictCurrency', methods=['POST'])
def predictCurrency():
    # Get the image from the request
    file= request.get_data()
    img = Image.open(io.BytesIO(file))

    # Preprocess the image
    img = img.resize((160, 160))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Make a prediction
    prediction = currency_model.predict(img_array)
    class_names = ["0.1", "0.2", "0.5","1","10","100","2","20","200","5","50"]
    predicted_class = class_names[np.argmax(prediction)]

    # Return the result
    result = {'class': predicted_class}
    return jsonify(result)





########################################################

if __name__ == '__main__':
    app.run(debug=True)
