import cv2
from preprocessing import PreprocessCars
import numpy as np 
class OdometerOCR:
    
    def __init__(self):
        self.preprocess_system = PreprocessCars()

    # def get_preprocessed_data(self, image, car_type='Mercedes_truck'):
    #     preprocess_func = getattr(self.preprocess_system, car_type, self.preprocess_system.default)
    #     processed_image, detected_texts = preprocess_func(image)
    #     return processed_image, detected_texts


    def get_preprocessed_data(self, image, car_type='Mercedes_truck'):
        # Get the preprocessing function from the preprocess_system based on car_type
        preprocess_func = getattr(self.preprocess_system, car_type, self.preprocess_system.default)
        # Call the preprocessing function and get the processed image and detections
        processed_image, detected_texts ,  last_trip_km, total_num_km = preprocess_func(image)
        # Return the processed image and detections
        return processed_image, detected_texts,  last_trip_km, total_num_km
    


    def draw_detections(self, image, detections):
        for bounding_box, _, _ in detections:
            points = np.array(bounding_box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)
        return image
