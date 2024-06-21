from ultralytics import YOLO
import cv2
from sort.sort import *
import img_utils

def video_detection(cap:cv2.VideoCapture):
    
    mot_tracker = Sort()

    model_detection = YOLO('./models/bestori_n.pt')
    names = model_detection.names

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
            if len(trucks) > 0:
                truck_ids = mot_tracker.update(np.asarray(trucks))
            
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
                        # crop_plate_thresh = img_utils.preprocess_image(crop_plate)
                        # cv2.imshow('thresh', crop_plate_thresh)
                        
                        # Baca teks plat nomor
                        plate_text, plate_text_conf = img_utils.read_plate_text(crop_plate)
                        
                        H, W, _ = crop_plate.shape

                        try:
                            cv2.putText(frame,
                                        plate_text,
                                        (int(x1-W), int(y1 - H)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        2,
                                        (0, 0, 255),
                                        2)
                        except:
                            pass
                        
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
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                else:
                    frame = img_utils.draw_border(frame, (int(x1),int(y1)), (int(x2), int(y2)))
        else:
            break
        frame = cv2.resize(frame, (1280,720))
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    img_utils.write_csv(result, 'output.csv')

    cap.release()
    cv2.destroyAllWindows()