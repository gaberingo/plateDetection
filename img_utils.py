def get_truck_id(license_plate, truck_ids):
    """Mengambil ID truk berdasarkan koordinat plat nomor

    Args:
        license_plate (tuple): _description_
        truck_ids (list): _description_

    Returns:
        tuple: Tuple berisi ID truk dan koordinat truk
    """
    return 0,0,0,0,0

def read_plate_text(plate_crop):
    """Baca teks plat nomor

    Args:
        plate_crop (numpy.ndarray): ndarray hasil preprocessing

    Returns:
        tuple: Tuple berisi teks plat nomor dan confidence
    """
    return '', 0