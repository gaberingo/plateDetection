import csv
import numpy as np
from scipy.interpolate import interp1d


def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    truck_ids = np.array([int(float(row['truck_id'])) for row in data])
    truck_bboxes = np.array([list(map(float, row['truck_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_truck_ids = np.unique(truck_ids)
    for truck_id in unique_truck_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['truck_id'])) == int(float(truck_id))]
        print(frame_numbers_, truck_id)

        # Filter data for a specific truck ID
        truck_mask = truck_ids == truck_id
        truck_frame_numbers = frame_numbers[truck_mask]
        truck_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = truck_frame_numbers[0]
        last_frame_number = truck_frame_numbers[-1]

        for i in range(len(truck_bboxes[truck_mask])):
            frame_number = truck_frame_numbers[i]
            truck_bbox = truck_bboxes[truck_mask][i]
            license_plate_bbox = license_plate_bboxes[truck_mask][i]

            if i > 0:
                prev_frame_number = truck_frame_numbers[i-1]
                prev_truck_bbox = truck_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_truck_bbox, truck_bbox)), axis=0, kind='linear')
                    interpolated_truck_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    truck_bboxes_interpolated.extend(interpolated_truck_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            truck_bboxes_interpolated.append(truck_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(truck_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['truck_id'] = str(truck_id)
            row['truck_bbox'] = ' '.join(map(str, truck_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['truck_id'])) == int(float(truck_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data


# Load the CSV file
with open('output.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'truck_id', 'truck_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('output_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
