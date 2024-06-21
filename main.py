import video_detection
import image_detection
import cv2

def main():
    cap = cv2.VideoCapture("./vid/vid2.mp4")
    video_detection.video_detection(cap)
    # img = cv2.imread('./img_test/img5.png')
    # image_detection.image_detection(img)

if __name__ == "__main__":
    main()
    