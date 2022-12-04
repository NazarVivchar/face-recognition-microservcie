import numpy as np
import cv2
from skimage import transform
import tensorflow as tf
import matplotlib.pyplot as plt
import base64

model = tf.keras.models.load_model('src/files/facenet_keras.h5')
image_size = 160


def convertImageStringToImage(image_str):
    im_bytes = base64.b64decode(image_str)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def prewhiten(x):
    mean = np.mean(x)

    std = np.std(x)

    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))

    y = np.multiply(np.subtract(x, mean), 1 / std_adj)

    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def align_image(image, margin):
    cascade = cv2.CascadeClassifier('src/files/haarcascade_frontalface_alt2.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=3,
                                     )

    faces_len = len(faces)

    if faces_len == 0:
        raise Exception(f"No people were detected on the image")

    biggest_face = faces[0]
    biggest_face_size = faces[0][2] * faces[0][3]

    for face in faces:
        face_size = face[2] * face[3]

        if face_size > biggest_face_size:
            biggest_face_size = face_size
            biggest_face = face

    (x, y, w, h) = biggest_face
    cropped = image[y - margin // 2:y + h + margin // 2,
              x - margin // 2:x + w + margin // 2, :]
    aligned = transform.resize(cropped, (image_size, image_size), mode='reflect')

    return aligned


def calc_embs(image_str, margin=10):
    image = convertImageStringToImage(image_str)
    plt.imshow(image)
    plt.show()
    aligned_image = prewhiten(align_image(image, margin))
    plt.imshow(aligned_image)
    plt.show()

    aligned_image = aligned_image.reshape(1, image_size, image_size, 3)
    pd = model.predict_on_batch(aligned_image)

    embs = l2_normalize(np.concatenate(pd))

    return embs.tolist()
