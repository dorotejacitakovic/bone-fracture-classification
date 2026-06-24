import os
from ultralytics import YOLO

print(os.listdir('/kaggle/input/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification'))

#Treniranje
model = YOLO('yolov8n-cls.pt')

results = model.train(
    data = '/kaggle/input/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification',
    epochs = 20,                    #broj prolazaka kroz foldere sa slikama
    imgsz = 224,                    #podesava se velicina slike koja se obradjuje
    batch = 16,                     #16 slika se istovremeno obradjuje
    project = "runs/fracture",      #gde ce se skladistiti rezultati
    name = "baseline"
)

#Export u .onnx
model.export(format='onnx', imgsz=224)

