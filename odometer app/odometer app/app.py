from flask import Flask, request, send_file, jsonify
from odometerocr import OdometerOCR
import numpy as np
import cv2
import io
import base64


app = Flask(__name__)
ocr_system = OdometerOCR()


# @app.route('/read-odometer', methods=['POST'])
# def read_odometer():

#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part'}), 400

#     file = request.files['image']

#     car_type = request.form.get('car_type', 'default')

#     if not file:
#         return jsonify({'error': 'No image selected for uploading'}), 400

#     # Convert the PIL Image to a NumPy array
#     in_memory_file = io.BytesIO()
#     file.save(in_memory_file)

#     data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
#     color_image_flag = 1
#     image = cv2.imdecode(data, color_image_flag)

#     processed_image, detections = ocr_system.get_preprocessed_data(image, car_type)
#     output_image = ocr_system.draw_detections(processed_image, detections)

#     # Encode the image to send as a response
#     _, buffer = cv2.imencode('.jpg', output_image)
#     response_image = io.BytesIO(buffer)

#     return send_file(response_image, mimetype='image/jpeg', as_attachment=True, attachment_filename='output.jpg')





# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)



@app.route('/read-odometer', methods=['POST'])
def read_odometer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    car_type = request.form.get('car_type', 'Mercedes_truck')
    if not file:
        return jsonify({'error': 'No image selected for uploading'}), 400

    # Convert the PIL Image to a NumPy array
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)

    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    processed_image, detections , last_trip_km, total_num_km = ocr_system.get_preprocessed_data(image, car_type)


    output_image = ocr_system.draw_detections(processed_image, detections)
    # Prepare the response

     
    # Encode the image to a base64 string i make this if i need to return the image in json file 
    # _, buffer = cv2.imencode('.jpg', output_image)
    # image_base64 = base64.b64encode(buffer).decode('utf-8')

    response = {
        # 'image': image_base64,
        # 'detections': detections,
         "last_trip_km" :  last_trip_km,
           "total_num_km":total_num_km
    }

    print(jsonify(response))
    # Display the output image with detections
    # cv2.imshow('Output Image', output_image)
    # cv2.waitKey(0)  # Wait for a key press to close the image window
    # cv2.destroyAllWindows()

    return jsonify(response)




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port =8000)
