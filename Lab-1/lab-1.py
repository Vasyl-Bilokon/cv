# import libraries
import cv2
import numpy as np

# read images for experiments
bright = cv2.imread("./pictures/IMG_1274.JPG")
dark = cv2.imread("./pictures/IMG_1275.JPG")

brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)

brightYCB = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
darkYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)

brightHSV = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
darkHSV = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)

# Display current images
cv2.imshow("bright BGR", bright)
cv2.imshow("bright HSV", brightHSV)
cv2.imshow("bright YCB", brightYCB)
cv2.imshow("bright LAB", brightLAB)

bgr = [40, 158, 16]
trash = 40

minBGR = np.array([bgr[0] - trash, bgr[1] - trash, bgr[2] - trash])
maxBGR = np.array([bgr[0] + trash, bgr[1] - trash, bgr[2] - trash])

maskBGR = cv2.inRange(bright, minBGR, maxBGR)
resultBGR = cv2.bitwise_and(bright, bright, mask = maskBGR)

# Convert 1D to 3D array and convert to HSV
hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]

minHSV = np.array([hsv[0] - trash, hsv[1] - trash, hsv[2] - trash])
maxHSV = np.array([hsv[0] + trash, hsv[1] + trash, hsv[2] + trash])

maskHSV = cv2.inRange(brightHSV, minHSV, maxHSV)
resultHSV = cv2.bitwise_and(brightHSV, brightHSV, mask=maskHSV)

# Convert 1D to 3D array and convert to YCrBb
ycb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]

minYCB = np.array([ycb[0] - trash, ycb[1] - trash, ycb[2] - trash])
maxYCB = np.array([ycb[0] + trash, ycb[1] + trash, ycb[2] + trash])

maskYCB = cv2.inRange(brightYCB, minYCB, maxYCB)
resultYCB = cv2.bitwise_and(brightYCB, brightYCB, mask=maskYCB)

# Convert 1D to 3D array and convert to LAB
lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
minLAB = np.array([lab[0] - trash, lab[1] - trash, lab[2] - trash])
maxLAB = np.array([lab[0] + trash, lab[1] + trash, lab[2] + trash])

maskLAB = cv2.inRange(brightLAB, minLAB, maxLAB)
resultLAB = cv2.bitwise_and(brightLAB, brightLAB, mask=maskLAB)

# Display results
cv2.imshow("Result BGR", resultBGR)
cv2.imshow("Result HSV", resultHSV)
cv2.imshow("Result YCB", resultYCB)
cv2.imshow("Result LAB", resultLAB)
cv2.waitKey(0)
cv2.destroyAllWindows()
