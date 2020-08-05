import tensorflow as tf

def get_model(classes):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.applications.InceptionResNetV2(
#            include_top=False,
            classes=classes,
            input_shape=(224,224,3)
            weights=None)
    return model

if __name__ == '__main__':
    face_recognition_model = get_model()
    print(face_recognition_model.summary())
