import os
import cv2

def crop_face(dataPath, savePath, target_size, cascPath='haarcascade_frontalface_default.xml'):

    # find a face in image and crop it
    faceCascade = cv2.CascadeClassifier(cascPath)
    files = os.listdir(dataPath)
    imgs_to_process = len(files)
    imgs_cropped = 0

    # create target directory if it doesn't exist
    os.makedirs(savePath, exist_ok=True)

    # Read source images
    for i, fn in enumerate(files):
        if i % 1000 == 0:
            print(f"Processing {i} of {imgs_to_process}: {fn}")

        # skip if image is already processed
        if os.path.isfile(os.path.join(savePath, fn)):
            continue

        try:
            image = cv2.imread(os.path.join(dataPath, fn))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 5)
        except:
            pass

        if len(faces) != 1:
            pass
        else:
            x, y, w, h = faces[0]
            image_crop = image[y: y+w, x : x+w, :]
            image_resize = cv2.resize(image_crop, target_size)
            cv2.imwrite(os.path.join(savePath, fn), image_resize)
            imgs_cropped+=1

    print(f"Cropped {imgs_cropped} faces from {imgs_to_process} in total")

if __name__ == "__main__":
    crop_face("dataset/img_align_celeba", "celeba/28by28", (28,28))