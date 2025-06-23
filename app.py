from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import zipfile
from datetime import datetime
from PIL import Image
import torch
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = Flask(__name__)
CORS(app)

# Setup model Real-ESRGAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model 4x
model_4x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler_4x = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model_4x,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

# Model Anime 6x
model_6x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
upsampler_6x = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus_anime_6B.pth',
    model=model_6x,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

# Mapping format input to valid Pillow format
format_mapping = {
    'JPG': 'JPEG',
    'JPEG': 'JPEG',
    'PNG': 'PNG',
    'WEBP': 'WEBP'
}

@app.route('/')
def home():
    return 'AI Image Upscaler with Real-ESRGAN is running!'

@app.route('/api/upscale', methods=['POST'])
def upscale_images():
    try:
        files = request.files.getlist('images')
        output_format = request.form.get('format', 'PNG').upper()
        scale = int(request.form.get('scale', 4))
        quality = int(request.form.get('quality', 95))

        valid_formats = ['PNG', 'JPG', 'JPEG', 'WEBP']
        if output_format not in valid_formats:
            return jsonify({'error': 'Invalid output format'}), 400

        if scale not in [2, 4, 6, 8]:
            return jsonify({'error': 'Invalid scale. Only 2x, 4x, 6x, and 8x are supported.'}), 400

        # Pilih model sesuai scale
        if scale <= 4:
            upsampler = upsampler_4x
        else:
            upsampler = upsampler_6x

        # Map to valid Pillow format
        pillow_format = format_mapping[output_format]

        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        if len(files) == 1:
            file = files[0]
            img = Image.open(file.stream).convert('RGB')
            img_np = np.array(img)

            output, _ = upsampler.enhance(img_np, outscale=scale)

            upscaled_img = Image.fromarray(output)
            img_bytes = io.BytesIO()

            if pillow_format == 'JPEG':
                upscaled_img.save(img_bytes, format=pillow_format, quality=quality)
            else:
                upscaled_img.save(img_bytes, format=pillow_format)

            img_bytes.seek(0)

            filename = f"upscaled-{datetime.now().strftime('%Y%m%d%H%M%S')}.{output_format.lower()}"
            return send_file(img_bytes, mimetype=f'image/{output_format.lower()}', as_attachment=True, download_name=filename)

        # Multi-file -> return ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            for idx, file in enumerate(files):
                img = Image.open(file.stream).convert('RGB')
                img_np = np.array(img)

                output, _ = upsampler.enhance(img_np, outscale=scale)

                upscaled_img = Image.fromarray(output)
                img_bytes = io.BytesIO()

                if pillow_format == 'JPEG':
                    upscaled_img.save(img_bytes, format=pillow_format, quality=quality)
                else:
                    upscaled_img.save(img_bytes, format=pillow_format)

                img_bytes.seek(0)

                filename = f"upscaled-{datetime.now().strftime('%Y%m%d%H%M%S')}-{idx+1}.{output_format.lower()}"
                zipf.writestr(filename, img_bytes.read())

        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='upscaled-images.zip')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
