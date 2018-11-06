
# 对测试数据进行测试

# 对测试数据进行测试
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    '../../data/validation/',
    target_size=(150, 150),
    batch_size=32,
    shuffle=False,
    classes=['dogs', 'cats'],

    class_mode='categorical')
model = load_model('model_weight.h5')
scoreSeg=model.evaluate_generator(
    validation_generator,
    steps=1
)
print("Accuracy = ",scoreSeg)

