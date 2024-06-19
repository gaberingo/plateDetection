import string
import easyocr
import re

# Initialize the OCR reader
reader = easyocr.Reader(['id'], gpu=False)

# Plate number format regex
regex = r"^(?:([A-Z]{1,2})(\d{1,4}))([A-Z]{1,2})$"

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def get_truck_id(license_plate, truct_ids):
    """Mengambil ID truk berdasarkan koordinat plat nomor

    Args:
        license_plate (tuple): _description_
        truck_ids (list): _description_

    Returns:
        tuple: Tuple berisi ID truk dan koordinat truk
    """
    
    x1, y1, x2, y2, _, _ = license_plate
    
    for truct_id in truct_ids:
        xt1, yt1, xt2, yt2, truct_id = truct_id
        if x1 > xt1 and y1 > yt1 and x2 < xt2 and y2 < yt2:
            return xt1, yt1, xt2, yt2, truct_id
    return -1,-1,-1,-1,-1

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if re.search(regex, text):
        return True
    return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {
        0: dict_int_to_char, 
        1: dict_int_to_char, 
        4: dict_int_to_char, 
        5: dict_int_to_char, 
        6: dict_int_to_char,
        2: dict_char_to_int, 
        3: dict_char_to_int
    }
    
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

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

        # if license_complies_format(text):
            # return format_license(text), confidence
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
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'truct_id', 'truct_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for truct_id in results[frame_nmr].keys():
                print(results[frame_nmr][truct_id])
                if 'truct' in results[frame_nmr][truct_id].keys() and \
                   'license_plate' in results[frame_nmr][truct_id].keys() and \
                   'text' in results[frame_nmr][truct_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            truct_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][truct_id]['truct']['bbox'][0],
                                                                results[frame_nmr][truct_id]['truct']['bbox'][1],
                                                                results[frame_nmr][truct_id]['truct']['bbox'][2],
                                                                results[frame_nmr][truct_id]['truct']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][truct_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][truct_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][truct_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][truct_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][truct_id]['license_plate']['confidence'],
                                                            results[frame_nmr][truct_id]['license_plate']['text'],
                                                            results[frame_nmr][truct_id]['license_plate']['text_confidence'])
                            )
        f.close()