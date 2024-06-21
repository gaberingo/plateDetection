from ultralytics import YOLO
import cv2
import img_utils
import matplotlib.pyplot as plt

model_detection = YOLO('./models/bestori_n.pt')
names = model_detection.names

def image_detection(img):
    results = model_detection(img)[0]
    for detection in results.boxes.data.tolist():
        x1,y1,x2,y2,conf,cls = detection
        if names[int(cls)] == 'plate':
            crop_plate = img[int(y1):int(y2), int(x1):int(x2), :]
            
            # Image processing
            # crop_plate_refine = img_utils.preprocess_image(crop_plate)
            # plt.imshow(crop_plate_refine)
            # plt.show()
            
            crop_plate = cv2.resize(crop_plate, (640, 480))
            
            plate_text, plate_text_conf = img_utils.read_plate_text(crop_plate)
            
            print(plate_text, plate_text_conf)
            
            H, W, _ = crop_plate.shape
            try:
                cv2.putText(
                    img,
                    plate_text,
                    (int(x1-W), int(y1-H)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    12
                    )
            except:
                pass
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        else:
            img = img_utils.draw_border(img, (int(x1),int(y1)), (int(x2), int(y2)))
    img = cv2.resize(img, (1280,720))
    plt.imshow(img)
    plt.show()