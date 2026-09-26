import cv2
import numpy as np


def video_crop(video_frame, type):
    l = video_frame.shape[0]
    new_frame = []
    for i in range(l):
        img = cv2.resize(video_frame[i], dsize=(340, 256))
        new_frame.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = np.array(new_frame)
    if type == 0:
        img = img[:, 16:240, 58:282, :]
    elif type == 1:
        img = img[:, :224, :224, :]
    elif type == 2:
        img = img[:, :224, -224:, :]
    elif type == 3:
        img = img[:, -224:, :224, :]
    elif type == 4:
        img = img[:, -224:, -224:, :]
    elif type == 5:
        img = img[:, 16:240, 58:282, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    elif type == 6:
        img = img[:, :224, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    elif type == 7:
        img = img[:, :224, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    elif type == 8:
        img = img[:, -224:, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    elif type == 9:
        img = img[:, -224:, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)

    return img


def image_crop(image, type):
    img = cv2.resize(image, dsize=(340, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if type == 0:
        img = img[16:240, 58:282, :]
    elif type == 1:
        img = img[:224, :224, :]
    elif type == 2:
        img = img[:224, -224:, :]
    elif type == 3:
        img = img[-224:, :224, :]
    elif type == 4:
        img = img[-224:, -224:, :]
    elif type == 5:
        img = img[16:240, 58:282, :]
        img = cv2.flip(img, 1)
    elif type == 6:
        img = img[:224, :224, :]
        img = cv2.flip(img, 1)
    elif type == 7:
        img = img[:224, -224:, :]
        img = cv2.flip(img, 1)
    elif type == 8:
        img = img[-224:, :224, :]
        img = cv2.flip(img, 1)
    elif type == 9:
        img = img[-224:, -224:, :]
        img = cv2.flip(img, 1)

    return img
