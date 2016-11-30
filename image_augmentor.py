import numpy as np
import cv2


def normalize_image(image):
    return cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def random_translate(image):
    rows, cols = image.shape[:2]
    x = np.random.randint(1,7)
    y = np.random.randint(1,7)
    trans_M = np.float32([[1,0,x],[0,1,y]])
    dst = cv2.warpAffine(image,trans_M,(cols,rows))
    dst = cv2.resize(dst[y:,x:],(48, 48))
    return dst


def random_rotate(image):
    rows, cols = image.shape[:2]
    rot_angle = np.random.randint(0,90)
    rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),rot_angle,1)
    dst = cv2.warpAffine(image,rot_M,(cols,rows))
    return dst


def random_brightness(image):
    dst = image + np.random.randint(-50,50)
    return cv2.normalize(dst, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def random_shear(image):
    rows, cols = image.shape[:2]
    pts1 = np.float32([[3,3],[15,3],[3,15]])
    x = 3+np.random.randint(1,5)
    y = 15+np.random.randint(1,5)
    pts2 = np.float32([[x,3],[y,x],[3,y]])
    
    shear_M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(image,shear_M,(cols,rows))
    return dst


def preturb_image(image):
    random_seed = np.random.ranf()
    if random_seed <= 0.2:
        dst = random_rotate(image)
    elif 0.2 < random_seed >= 0.4:
        dst = random_translate(image)
    elif 0.4 < random_seed >= 0.6:
        dst = random_brightness(image)
    elif 0.6 < random_seed >= 0.8:
        dst = random_shear(image)
    else:
        dst = image
    return dst


def preturb_images(images):
    for i,image in enumerate(images):
        # images[i,:,:,:] = normalize_image(image)
        images[i,:,:,:] = preturb_image(image)
    return images


def batch_generator(images, labels, batch_count=10,batch_size=50):
    idxs = np.arange(images.shape[0])
    for i in range(batch_count):
        batch_idx = np.random.choice(idxs, size=batch_size, replace=False)
        batch_images = images[batch_idx]
        batch_images = preturb_images(batch_images)
        batch_labels = labels[batch_idx]
        yield batch_images, batch_labels
        
if __name__ == '__main__':
    from six.moves import cPickle as pickle
    testing_file = 'test.p'
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    
    X_test, y_test = test['features'], test['labels']
    for x,y in batch_generator(X_test, y_test):
        print(y)
