
# 对测试数据进行测试
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
    validation_generator
)
print("Accuracy = ",scoreSeg)
# loss_and_metrics = model.evaluate(X_valid, y_valid, batch_size=128)