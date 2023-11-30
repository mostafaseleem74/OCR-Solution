import cv2
from paddleocr import PaddleOCR

class PreprocessCars:
    def __init__(self):
        self.ocr = PaddleOCR(use_gpu=True)

    # def read_odometer(self, image):
    #     results = self.ocr.ocr(image)
    #     detections = []
    #     if results:
    #         for result in results:
    #             for detection in result:
    #                 bounding_box = detection[0]
    #                 text, confidence = detection[1]
    #                 detections.append((bounding_box, text, confidence))
    #     return image, detections

    def ensure_dot_and_space_before_last_char(self , text):
        
        # Check if the character before the last is not a dot
        if text[-2] != ".":
            # If it's not a dot, add a dot before the last character
            text = text[:-1] + "." + text[-1]
        # Return the text concatenated with " km"
        return text + " km"





    def read_odometer(self, image):
        results = self.ocr.ocr(image)
        detections = []
        last_trip_km = None
        total_num_km = None
        
        if results:
            for result in results:
                for detection in result:
                    bounding_box = detection[0]
                    text, confidence = detection[1]
                    detections.append((bounding_box, text, confidence))
        
        #  sure there are enough detections 
        # if len(detections) > 11:
        last_trip_km = detections[8][1] 
        total_num_km = detections[9][1]  

        last_trip_km=self.ensure_dot_and_space_before_last_char(last_trip_km)
        total_num_km=self.ensure_dot_and_space_before_last_char(total_num_km)

        return image, detections, last_trip_km, total_num_km


    def default(self, image):
        print("\n_____ default processing _____\n")
        return self.read_odometer(image)


    def Mercedes_truck(self, image):
        # Custom preprocessing for Mercedes truck type
        # processed_image = self.some_preprocessing_method_for_mercedes_truck(image)
        print("\n_____Processing for Mercedes truck_____\n")
        # You should return the processed image as well along with the OCR results
        # processed_image, detected_texts , last_trip_km, total_num_km= self.read_odometer(image)
        # return processed_image, detected_texts , last_trip_km, total_num_km
        return self.read_odometer(image)

    

    def car_type_2(self, image):
        # Custom preprocessing for car type 2
        processed_image = self.some_preprocessing_method_2(image)
        return self.read_odometer(processed_image)

    # ... more car types ...

    # Example preprocessing methods (placeholders)
    def some_preprocessing_method_1(self, image):
        # Implement actual preprocessing for car type 1
        return image

    def some_preprocessing_method_2(self, image):
        # Implement actual preprocessing for car type 2
        return image

# ... add more preprocessing methods as needed ...
