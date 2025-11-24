import os
from flask import Flask, render_template, request, redirect
from inference import get_prediction
from commons import transform_image

from google.colab.output import eval_js
from pyngrok import ngrok  # must install pyngrok

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return redirect(request.url)

        img_bytes = file.read()
        file_tensor = transform_image(image_bytes=img_bytes)
        class_name = get_prediction(file_tensor)

        return render_template('result.html', class_name=class_name)

    return render_template('index.html')

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(" * Public URL:", public_url)
    app.run(port=5000)
