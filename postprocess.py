import cv2
import numpy as np


def visualize(img, mask):
    mask = cv2.medianBlur(mask,3)
    mask = cv2.GaussianBlur(mask,(3,3), 0)
    ret,mask = cv2.threshold(mask,0.5,255,cv2.THRESH_BINARY)
    output_mask = mask.copy()

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask[:, :, 0] = 0
    mask[:, :, 1] = 0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)
    return output_mask, overlay

segmentation = cv2.imread("output_2.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("output_0.png")
mask, overlay = visualize(img, segmentation)


cv2.imwrite(f"output_postprocessed.png", mask)
cv2.imwrite(f"output_overlay.png", overlay)