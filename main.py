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

while ret and frm_num < 250:
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
                
        truct_ids = mot_tracker.update(np.array(trucks))
        
        # Deteksi Plat Nomor
        for plate in results.boxes.data.tolist():
            x1,y1,x2,y2,conf,cls = plate
            if names[int(cls)] == 'plate':
                
                # Mengambil ID truk berdasarkan koordinat plat nomor
                xt1, yt1, xt2, yt2, truct_id = img_utils.get_truck_id(plate, truct_ids)
                
                if truct_id != -1:
                    # Crop plat nomor
                    crop_plate = frame[int(y1):int(y2), int(x1):int(x2), :]
                    
                    # Image processing plat nomor
                    crop_plate_gray = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2GRAY)
                    crop_plate_thresh = cv2.threshold(crop_plate_gray, 75, 255, cv2.THRESH_BINARY_INV)[1]
                    
                    # Baca teks plat nomor
                    plate_text, plate_text_conf = img_utils.read_plate_text(crop_plate_thresh)
                    
                    # Mencatat hasil
                    if plate_text is not None:
                        result[frm_num][truct_id] = {
                            'truct':{'bbox':[xt1, yt1, xt2, yt2]},
                            'license_plate':{
                                'bbox':[x1,y1,x2,y2],
                                'text':plate_text,
                                'confidence':conf,
                                'text_confidence':plate_text_conf,}
                            }
    else:
        break
    
img_utils.write_csv(result, 'output.csv')

# cap.release()
# cv2.destroyAllWindows()