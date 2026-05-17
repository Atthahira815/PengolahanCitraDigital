import cv2
import numpy as np

class RealTimeEnhancement:

    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.history_buffer = []

    def enhance_frame(self, frame, enhancement_type='adaptive'):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if enhancement_type == 'adaptive':

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

        elif enhancement_type == 'gamma':

            gamma = 1.5
            invGamma = 1.0/gamma
            table = np.array([(i/255.0)**invGamma *255
                    for i in np.arange(256)]).astype("uint8")

            enhanced = cv2.LUT(gray, table)

        else:
            enhanced = gray

        # convert kembali ke BGR agar bisa tampil di video
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return enhanced


# Webcam streaming
cap = cv2.VideoCapture(0)

enhancer = RealTimeEnhancement()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    enhanced_frame = enhancer.enhance_frame(frame,'adaptive')

    cv2.imshow("Original", frame)
    cv2.imshow("Enhanced Video", enhanced_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):   # keluar
        break
    elif key == ord('s'): # save frame
        cv2.imwrite("frame.jpg", enhanced_frame)

cap.release()
cv2.destroyAllWindows()