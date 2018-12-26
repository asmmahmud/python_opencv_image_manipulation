import cv2

cap = cv2.VideoCapture("../resources/SampleVideo_1280x720_1mb.mp4.mp4")
print(cap.isOpened())  # True = read video successfully. False - fail to read video.

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_video.avi", fourcc, 20.0, (640, 360))
print(out.isOpened())  # True = write out video successfully. False - fail to write out video.

cap.release()
out.release()
