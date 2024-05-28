import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image


class CartoonGen:
    def __init__(self, model_no: int):
        self.pic_form = ['.jpeg','.jpg','.png','.JPEG','.JPG','.PNG']
        device_name = ort.get_device()

        if device_name == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device_name == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        model = {
            1: 'AnimeGANv2_Shinkai',
            2: 'AnimeGANv2_Paprika',
            3: 'AnimeGANv2_Hayao',
            4: 'AnimeGAN_Hayao'
        }
        self.session = ort.InferenceSession(f'./models/{model[model_no]}.onnx', providers=providers)

    def process_image(self,img, x32=True):
        h, w = img.shape[:2]
        if x32: # resize image to multiple of 32s
            def to_32s(x):
                return 256 if x < 256 else x - x%32
            img = cv2.resize(img, (to_32s(w), to_32s(h)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
        return img

    def load_test_data(self,image):
        # img0 = cv2.imread(path).astype(np.float32)
        image_np = np.array(image)
        
        # Convert the image to BGR format if it's not already in that format
        if image_np.shape[2] == 4:  # If the image has an alpha channel
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif image_np.shape[2] == 3:  # If the image is RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Convert the image to float32
        img0 = image_np.astype(np.float32)
            
        img = self.process_image(img0)
        img = np.expand_dims(img, axis=0)
        return img, img0.shape[:2]

    def Convert(self,img, scale):
        x = self.session.get_inputs()[0].name
        y = self.session.get_outputs()[0].name

        fake_img = self.session.run(None, {x : img})[0]
        images = (np.squeeze(fake_img) + 1.) / 2 * 255
        images = np.clip(images, 0, 255).astype(np.uint8)

        output_image = cv2.resize(images, (scale[1],scale[0]))
        return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
# img = "prafull.jpg"
# obj = CartoonGen(4)

# mat, scale = obj.load_test_data(img)
# res = obj.Convert(mat, scale)

# cv2.imwrite("cartoon.jpg", res)

# i = np.array(Image.open("prafull.jpg"), dtype=np.float32)
# print(i)