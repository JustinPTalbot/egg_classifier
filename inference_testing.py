from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import image_utils, ImageDataGenerator
import numpy as np
import os

model = load_model('egg_classifier.h5')

for filename in os.listdir('predict'):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image
        img_path = os.path.join('predict', filename)
        img = image.image_utils.load_img(img_path, target_size=(150, 150))
        x = image_utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Rescale pixel values to 0-1 range

        # Make the prediction
        preds = model.predict(x)
        pred_class = np.argmax(preds)
        class_dict = {0: 'cracked', 1: 'empty', 2: 'good'} # Your class dictionary
        pred_label = class_dict[pred_class]
        
        # Print the prediction
        print(f'{filename} --> {pred_label}')
        


# test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
#         'test',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='categorical')

# # {'crack': 0, 'empty': 1, 'good': 2}