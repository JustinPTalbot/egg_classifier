'''
Dataset Citation:
    @misc{frank pereny_2023,
	title={Broken Eggs},
	url={https://www.kaggle.com/dsv/5367524},
	DOI={10.34740/KAGGLE/DSV/5367524},
	publisher={Kaggle},
	author={Frank Pereny},
	year={2023}
}
'''


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

# Set up the image data generator for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('train',
                                               target_size=(150, 150),
                                               batch_size=32,
                                               class_mode='categorical')

test_data = test_datagen.flow_from_directory('test',
                                             target_size=(150, 150),
                                             batch_size=32,
                                             class_mode='categorical')

# Build the model1
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(train_data,
          steps_per_epoch=train_data.samples//train_data.batch_size,
          validation_data=test_data,
          validation_steps=test_data.samples//test_data.batch_size,
          epochs=20)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data)

# Print the test loss and accuracy
print('Model:')
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Save the model
model.save('model.h5')

