from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt
import sys

def plot_bboxes(r, names):
    annotator = Annotator(r.orig_img)
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]
        c = box.cls
        annotator.box_label(b, names[int(c)])
    img = annotator.result() 
    return img


def detect_logo(image_path):
    model = YOLO('yolo_models/best_vk_avito.pt')
    names = model.names
    result = model(image_path)
    img = plot_bboxes(result[0], names)
    for c in result[0].boxes.cls:
        print('Был найден логотип компании:',names[int(c)])
    plt.imsave('yolo_predict.png', img)


if __name__ == "__main__":
    image_path = sys.argv[1]
    prediction = detect_logo(image_path)
    print(f"Результат обработки сохранен в файле 'yolo_predict.png' ")