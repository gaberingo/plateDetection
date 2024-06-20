from ultralytics import YOLO
import cv2
from sort.sort import *
import img_utils

mot_tracker = Sort()

model_detection = YOLO('/home/witech/OdooWitech/PlateDetection/models/bestv4_n.pt')
names = model_detection.names

cap = cv2.VideoCapture('./vid/vid1.mp4')

ret = True
frm_num = -1

result = {}

while ret:
    rep, frame = cap.read()
    frm_num += 1
    result[frm_num] = {}
    if rep:
        # Deteksi Truk
        results = model_detection(frame)[0]
        trucks = []
        for detection in results.boxes.data.tolist():
            x1,y1,x2,y2,conf,cls = detection
            if names[int(cls)] == 'truck':
                trucks.append([x1,y1,x2,y2,conf])
                
        truck_ids = mot_tracker.update(np.array(trucks))
        
        # Deteksi Plat Nomor
        for plate in results.boxes.data.tolist():
            x1,y1,x2,y2,conf,cls = plate
            if names[int(cls)] == 'plate':
                
                # Mengambil ID truk berdasarkan koordinat plat nomor
                xt1, yt1, xt2, yt2, truck_id = img_utils.get_truck_id(plate, truck_ids)
                
                if truck_id != -1:
                    # Crop plat nomor
                    crop_plate = frame[int(y1):int(y2), int(x1):int(x2), :]
                    
                    # Image processing
                    crop_plate_thresh = img_utils.preprocess_image(crop_plate)
                    cv2.imshow('thresh', crop_plate_thresh)
                    # grayscale_crop_plate = cv2.resize(crop_plate, (crop_plate.shape[1]*3, crop_plate.shape[0]*3))
                    # grayscale_crop_plate = img_utils.sharpening_font(grayscale_crop_plate)
                    # grayscale_crop_plate = cv2.cvtColor(grayscale_crop_plate, cv2.COLOR_BGR2GRAY)
                    
                    
                    # cv2.imshow('grayscale', grayscale_crop_plate)
                    # crop_plate_thresh = cv2.adaptiveThreshold(grayscale_crop_plate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)
                    # crop_plate_thresh = img_utils.noise_removal(crop_plate_thresh)
                    # crop_plate_thresh = img_utils.thin_font(crop_plate_thresh)
                    # cv2.imshow('thresh', crop_plate_thresh)
                    
                    # Baca teks plat nomor
                    plate_text, plate_text_conf = img_utils.read_plate_text(crop_plate_thresh)
                    
                    # Mencatat hasil
                    if plate_text is not None:
                        result[frm_num][truck_id] = {
                            'truck':{'bbox':[xt1, yt1, xt2, yt2]},
                            'license_plate':{
                                'bbox':[x1,y1,x2,y2],
                                'text':plate_text,
                                'confidence':conf,
                                'text_confidence':plate_text_conf,}
                            }
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
img_utils.write_csv(result, 'output.csv')

cap.release()
cv2.destroyAllWindows()