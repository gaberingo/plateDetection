import easyocr
import cv2
import numpy as np
# Initialize the OCR reader
reader = easyocr.Reader(['id'], gpu=False)

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
        print(detection)
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
        
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img
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
    # Resize image
    image = cv2.resize(image, (640, 480))
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Edge detection using Canny
    # image = cv2.Canny(image, 50, 150)

    # Morphological operations (dilation and erosion)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed", closed)
    
    # Adaptive thresholding
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Crop the image to the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    plate_image = image[y:y + h, x:x + w]
    
    return plate_image
