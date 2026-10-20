import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('mahdbaseModel.keras')

img = cv2.imread('sample12.jpeg', cv2.IMREAD_GRAYSCALE)

invImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 10)

brush = np.ones((3,3), np.uint8)

invImg = cv2.dilate(invImg, brush, iterations=1)

contours, _ = cv2.findContours(invImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

X = []
locations = []

for hill in contours:
    x, y, w, h = cv2.boundingRect(hill)

    number = invImg[y:y+h, x:x+w]

    if (h < 6 or w < 6):
        continue

    locations.append((x, y))

    if (h > w):

        newH = 20
        newW = max(1, int((w / h) * 20))
    else:
        newW = 20
        newH = max(1, int((h / w) * 20))

    resizedNum = cv2.resize(number, (newW, newH), interpolation=cv2.INTER_AREA)

    topPadding = (28 - newH) // 2
    bottomPadding = (28 - newH - topPadding)

    leftPadding = (28 - newW) // 2
    rightPadding = (28 - newW - leftPadding)

    finalNum = cv2.copyMakeBorder(resizedNum, topPadding, bottomPadding, leftPadding, rightPadding, cv2.BORDER_CONSTANT, value=0)

    m = cv2.moments(finalNum)

    print(f"Box weight = {m["m00"]}/n")
    
    wX = m["m10"] / m["m00"]
    wY = m["m01"] / m["m00"]
        
    shiftX = 14.0 - wX
    shiftY = 14.0 - wY
        
    M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        
    finalNum = cv2.warpAffine(finalNum, M, (28, 28))

    finalNum = (finalNum / 255)
    X.append(finalNum)

    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 0), 1)

print(f"I see {len(X)} numbers")

X = np.array(X)

predictions = model.predict(X, verbose = 0)
predictedY = np.argmax(predictions, axis=1)

for i in range(len(predictedY)):

    x, y = locations[i]

    cv2.putText(img, str(predictedY[i]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

cv2.imshow("Detected Numbers", invImg)
cv2.imwrite("output.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()