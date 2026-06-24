import cv2
import numpy as np
import os

#Ucitavanje ONNX modela kroz OpenCV DNN
net = cv2.dnn.readNetFromONNX('best.onnx')

classes = ['fractured', 'not fractured']

#Preprocessing funkcije
#CLAHE - poboljsavanje kontrasta
def clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


#GAUSS - uklanjanje suma
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


#CANNNY - detekcija ivica
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


#CLAHE + GAUSS
def clahe_gauss(image):
    image = clahe(image)
    image = gauss(image)

    return image

#INFERENCIJA
def predict(img, label = ""):
    #blob - binary large object
    blob = cv2.dnn.blobFromImage(
        img, 
        scalefactor = 1/255.0,          #YOLO uzima vrednosti od 0 do 1
        size = (224, 224),              #originalna slika je u zadatom opsegu
        mean = (0, 0, 0),               #prosek kanala boja
        swapRB = True                   #BGR model(blue, green, red)
    )
    #Propustanje kroz mrezu(interferencija)
    net.setInput(blob)
    output = net.forward()          #izlaz matrica(1, 2)

    #Softmax funkcija
    results = output[0]                            
    exp_results = np.exp(results - np.max(results))
    verovatnoca = exp_results / exp_results.sum()           #rezultati izmedju 0 i 1(u zbiru daju 1)

    prediction = classes[np.argmax(verovatnoca)]
    confidence = np.max(verovatnoca)

    print(f"[{label}] {prediction} ({confidence*100:.1f}%)")
    return prediction, confidence


def test_preprocessing(image_path, true_label):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Not possible to open {image_path}")
        return
    
    print(f"Image: {os.path.basename(image_path)} | True class: {true_label}")

    predict(img, label = "No preprocessing")
    predict(clahe(img), label = "CLAHE")
    predict(gauss(img), label = "GAUSSIAN BLUR")
    predict(canny(img), label = "CANNY")
    predict(clahe_gauss(img), label = "CLAHE + GAUSS")


test_img = 'test_image'
correct = {'baseline': 0, 'clahe': 0, 'gauss': 0, 'canny': 0, 'clahe_gauss': 0}
sum = 0

for klasa in os.listdir(test_img):
    klasa_dir = os.path.join(test_img, klasa)
    if not os.path.isdir(klasa_dir):
        continue
    slike = os.listdir(klasa_dir)

    for img in slike:
        path = os.path.join(klasa_dir, img)
        image = cv2.imread(path)
        if image is None:
            continue

        sum += 1
        print(f"\nImage: {img} | True class: {klasa}")

        p1, procenat1 = predict(image, label = "No preprocessing")
        p2, procenat2 = predict(clahe(image), label = "CLAHE")
        p3, procenat3 = predict(gauss(image), label = "GAUSSIAN BLUR")
        p4, procenat4 = predict(canny(image), label = "CANNY")
        p5, procenat5 = predict(clahe_gauss(image), label = "CLAHE + GAUSS")

        if p1 == klasa:
            correct['baseline'] += 1
        if p2 == klasa:
            correct['clahe'] += 1
        if p3 == klasa:
            correct['gauss'] += 1
        if p4 == klasa:
            correct['canny'] += 1
        if p5 == klasa:
            correct['clahe_gauss'] += 1

print("-"*50)
print("TACNOST PO METODAMA:\n")

for metoda, tacno in correct.items():
    print(f"{metoda}: {tacno}/{sum} ({tacno/sum*100:.1f})%")