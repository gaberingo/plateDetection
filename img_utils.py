import easyocr
import cv2
import numpy as np
# Initialize the OCR reader
reader = easyocr.Reader(['id'], gpu=False)

# Plate number format regex
# regex = r"^(?:([A-Z]{1,2})(\d{1,4}))([A-Z]{1,2})$"

def get_truck_id(license_plate, truck_ids):
    """Mengambil ID truk berdasarkan koordinat plat nomor

    Args:
        license_plate (tuple): _description_
        truck_ids (list): _description_

    Returns:
        tuple: Tuple berisi ID truk dan koordinat truk
    """
    
    x1, y1, x2, y2, _, _ = license_plate
    
    for truck_id in truck_ids:
        xt1, yt1, xt2, yt2, truck_id = truck_id
        if x1 > xt1 and y1 > yt1 and x2 < xt2 and y2 < yt2:
            return xt1, yt1, xt2, yt2, truck_id
    return -1,-1,-1,-1,-1

def read_plate_text(plate_crop):
    """Baca teks plat nomor

    Args:
        plate_crop (numpy.ndarray): ndarray hasil preprocessing

    Returns:
        tuple: Tuple berisi teks plat nomor dan confidence
    """
    
    detections = reader.readtext(plate_crop)
    for detection in detections:
        _, text, confidence = detection
        
        text = text.upper().replace(' ', '')
        return text, confidence
    
    return None, None

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'truck_id', 'truck_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for truck_id in results[frame_nmr].keys():
                print(results[frame_nmr][truck_id])
                if 'truck' in results[frame_nmr][truck_id].keys() and \
                   'license_plate' in results[frame_nmr][truck_id].keys() and \
                   'text' in results[frame_nmr][truck_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            truck_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][truck_id]['truck']['bbox'][0],
                                                                results[frame_nmr][truck_id]['truck']['bbox'][1],
                                                                results[frame_nmr][truck_id]['truck']['bbox'][2],
                                                                results[frame_nmr][truck_id]['truck']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][truck_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][truck_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][truck_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][truck_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][truck_id]['license_plate']['confidence'],
                                                            results[frame_nmr][truck_id]['license_plate']['text'],
                                                            results[frame_nmr][truck_id]['license_plate']['text_confidence'])
                            )
        f.close()
        
# Noise Reduction
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

# Dilation and Erosion
def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

# Sharpening image
def sharpening_font(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def preprocess_image(image):
    # Step 0: Resize image
    image = cv2.resize(image, (image.shape[1]*3, image.shape[0]*3))
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Step 3: Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Step 4: Morphological operations (dilation and erosion)
    # kernel = np.ones((3, 3), np.uint8)
    # closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Step 5: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Step 6: Find contours
    # contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return thresh
