import cv2
import numpy as np

gt = cv2.imread("output_0.png")
gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)


gt_edge = cv2.Canny(gt, 25, 100)
kernel = np.ones((3,3), np.uint8)
gt_edge = cv2.dilate(gt_edge, kernel, iterations=1)
gt_edge = cv2.erode(gt_edge, kernel, iterations=1)
cv2.imwrite("edge.png", gt_edge)

img = cv2.imread("output_2.png", cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img,3)
img = cv2.GaussianBlur(img,(3,3), 0)
ret,img = cv2.threshold(img,0.5,255,cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)
#img = cv2.erode(img, kernel, iterations=1)
#img = cv2.dilate(img, kernel, iterations=1)
 

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_color[:, :, 0] = 0
img_color[:, :, 1] = 0
overlay = cv2.addWeighted(gt, 1, img_color, 0.5, 0.0)

cv2.imwrite(f"output_postprocessed.png", img)
cv2.imwrite(f"output_overlay.png", overlay)