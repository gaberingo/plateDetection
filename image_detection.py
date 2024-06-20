from ultralytics import YOLO
import cv2
import img_utils

model_detection = YOLO('./models/bestv4_n.pt')
names = model_detection.names

def image_detection(img):
    results = model_detection(img)[0]
    for detection in results.boxes.data.tolist():
        x1,y1,x2,y2,conf,cls = detection
        if names[int(cls)] == 'plate':
            crop_plate = img[int(y1):int(y2), int(x1):int(x2), :]
            
            # Image processing
            # crop_plate_refine = img_utils.preprocess_image(crop_plate)
            # cv2.imshow('thresh', crop_plate_refine)
            
            plate_text, plate_text_conf = img_utils.read_plate_text(crop_plate)
            print(plate_text, plate_text_conf)
            H, W, _ = crop_plate.shape

            try:
                # img[int(y1) - H - 100:int(y1) - 100, int((x2 + x1 - W) / 2):int((x2 + x1 + W) / 2), :] = crop_plate

                # img[int(y1) - H - 400:int(y1) - H - 100, int((x2 + x1 - W) / 2):int((x2 + x1 + W) / 2), :] = (255, 255, 255)

                (text_width, text_height), _ = cv2.getTextSize(
                    plate_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    17)

                cv2.putText(img,
                            plate_text,
                            (int((x2 + x1 - text_width) / 2), int(y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 0),
                            2)

            except:
                pass
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        else:
            img = img_utils.draw_border(img, (int(x1),int(y1)), (int(x2), int(y2)))
    img = cv2.resize(img, (640,480))
    cv2.imshow("Detection",img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()