from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from moviepy.editor import VideoFileClip
import cv2
import json
import numpy as np

from models import CartoonGen

from PIL import Image
import io

app = Flask(__name__)
CORS(app)


@app.route("/upload", methods=['post'])
def upload():
    image_file = request.files['image']
    file_name = image_file.filename

    img_ext = file_name.split(".")[1]

    # new_name = f"input.{file_name.split('.')[1]}"
    output_name = f"output.{image_file.filename.split('.')[1]}"

    json_data = json.loads(request.form['data'])
    model_num = int(json_data['model_num'])
    iterations = int(json_data['iterations'])
    print(model_num)

    if image_file:
        try:
            image = Image.open(io.BytesIO(image_file.read()))

            model = CartoonGen(model_num)
            current = image

            for _ in range(iterations):
                mat, scale = model.load_test_data(current)
                res = model.Convert(mat, scale)
                current = res

            cv2.imwrite(f"result.{img_ext}", current)

            return send_file(f"result.{img_ext}", mimetype=f'image/{img_ext}'), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route("/", methods=['get'])
def main():
    return "server"


if __name__ == "__main__":
    app.run(port=8000, debug=True)