import csv
import pandas as pd
import cv2
import numpy as np
from imutils import face_utils
import dlib
import joblib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
blue, green, red, yellow, purple = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX

def CalSpo22(video_file
             # , bvp_file
             # , sub_folder_path
             ):

    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    with open(f"文件路径.csv", 'w',
              newline=''
              ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10', 'var11', 'var12', 'var13', 'var14', 'var15', 'var16', "var17", "var18"]
            )
        while cap.isOpened():
            ret, frame = cap.read()
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_count += 1
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                # Get the Cheek ROI of Face获取面部的Cheek ROI
                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                start_face = (face.left(), face.top())
                end_face = (face.right(), face.bottom())
                start_cheekl = (landmarks[4][0], landmarks[29][1])
                end_cheekl = (landmarks[48][0], landmarks[33][1])
                start_cheekr = (landmarks[54][0], landmarks[29][1])
                end_cheekr = (landmarks[12][0], landmarks[33][1])
                start_cheekw = (landmarks[49][0], landmarks[7][1])
                end_cheekw = (landmarks[55][0], landmarks[11][1])

                cv2.rectangle(frame, start_face, end_face, green, 2)
                cv2.rectangle(frame, start_cheekl, end_cheekl, green, 1)
                cv2.rectangle(frame, start_cheekr, end_cheekr, green, 1)
                cv2.rectangle(frame, start_cheekw, end_cheekw, green, 1)
                Ka = []

                # Calculate Ka left cheek
                image = frame[start_cheekl[1]:end_cheekl[1], start_cheekl[0]:end_cheekl[0]]
                (B, G, R) = cv2.split(image)
                lDCB, lACB, lDCR, lACR, lDCG, lACG = np.mean(B), np.std(B), np.mean(R), np.std(R), np.mean(
                    G), np.std(G)

                # Calculate Ka right cheek
                image = frame[start_cheekr[1]:end_cheekr[1], start_cheekr[0]:end_cheekr[0]]
                (B, G, R) = cv2.split(image)
                rDCB, rACB, rDCR, rACR, rDCG, rACG = np.mean(B), np.std(B), np.mean(R), np.std(R), np.mean(
                    G), np.std(G)

                # Calculate Ka F cheek
                image = frame[start_face[1]:end_face[1], start_face[0]:end_face[0]]
                (B, G, R) = cv2.split(image)
                fDCB, fACB, fDCR, fACR, fDCG, fACG = np.mean(B), np.std(B), np.mean(R), np.std(R), np.mean(G), np.std(G)

                writer.writerow([lDCB, lACB, lDCR, lACR, lDCG, lACG, rDCB, rACB, rDCR, rACR, rDCG, rACG, fDCB, fACB, fDCR, fACR, fDCG, fACG])

        print(frame_count - 1)

def read(vdieo_path):

    #计算血氧
    CalSpo22(video_path)

    data2 = pd.read_csv(f'文件路径.csv')
    predictors = [
        'var1', 'var2', 'var3', 'var4', 'var5', 'var6',
        'var7', 'var8', 'var9', 'var10', 'var11', 'var12',
        'var13', 'var14', 'var15', 'var16', 'var17', 'var18',
    ]

    x_2 = data2[np.array(predictors)].values

    model = joblib.load('ExtraTreesRegressor.pkl')
    y_hat = model.predict(x_2)

    a = np.mean(y_hat)

    return a

if __name__ == '__main__':

    video_path = "指定路径"
    SPO2 = read(video_path)
    print("SPO2 value: {}".format(SPO2))