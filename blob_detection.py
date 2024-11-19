import cv2 
import numpy as np 

blobs = cv2.imread("blobs.jpeg", 0)
blob_detector = cv2.SimpleBlobDetector_Params()

#setting parameters for blob detection:
#minimum and maximum of intensity of pixel
#blob_detector.minThreshold = 10
#blob_detector.maxThreshold = 200
#minium and maximum area set in pixels
blob_detector.filterByArea = True
blob_detector.minArea = 60
#blob_detector.maxArea = 1000
blob_detector.filterByColor = True
blob_detector.blobColor = 0
blob_detector.filterByCircularity = True
blob_detector.minCircularity = 0.9
blob_detector.filterByConvexity = True
blob_detector.minConvexity = 0.2
blob_detector.filterByInertia = True
blob_detector.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(blob_detector)
key_blobs = detector.detect(blobs)
print(key_blobs)
number_of_blobs = len(key_blobs)

#adding text 
color = (183,54,4)
thickness = 2
font = cv2.FONT_HERSHEY_COMPLEX
fontscale = 3
origin = (10,100)
text = cv2.putText(blobs, str(number_of_blobs), origin, fontscale, font, color, thickness, cv2.LINE_AA)

kernel_size = np.zeros((1,1))
final_image = cv2.drawKeypoints(blobs, key_blobs, kernel_size, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#final_image = cv2.drawKeypoints(blobs, key_blobs, np.array([]), (153,56,94), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Final image", final_image)


cv2.waitKey(0)
cv2.destroyAllWindows()