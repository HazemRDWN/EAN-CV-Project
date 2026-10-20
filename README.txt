Overview:
This project aims to train a model that can identify handwritten eastern Arabic numerals written on a piece of paper and write software that does the appropriate preprocessing on the incoming image to allow the model to make an accurate prediction.

Dataset:
The dataset used to train the model was the MAHDBase from the AUC, which contains ~60,000 images of handwritten eastern Arabic numerals from different individuals for training, and ~10,000 images for testing. 
The data was stored into two folders, one for testing and one for training, and within each folder there were even more folder partitions because operating systems would struggle with having 60,000 files in one folder, this would complicate training and make it take longer due to the constant opening and closing of files.
The solution was to write code that went through all the folders and files only once, and store all the data in a single file (mahdbase.npz). That way when training, the model only opens that file once and reads from it directly.

Model:
An artificial neural network was used for the model, with four hidden layers each with a neuron density of 256 neurons. Few layers were used because the images are simple, white on black (after preprocessing) images, and 256 neurons per layer helped with the big number of features (pixels).
Training was done on the preprocessed MAHDBase data. Preprocessing included inverting the image so that the background is black and ink is white, that way most of the data sent to the model is zeros with only the relevant ink being high values, simplifying the math. Also normalizing the pixel values using min-max normalization since all pixel values are within the 0-255 range. Once the model is trained it is saved to be used in predictions later and remove the need for retraining when making each prediction.


Prediction Preprocessing:
The image is first made grayscale to eliminate color and help simplify further, then adaptive thresholding is used to eliminate shade and force each pixel to either be black (0) or white (255) with no in-between and invert the image to be like the training data. Adaptive thresholding helps achieve this in different lighting conditions by deciding the color of each pixel by looking at a certain sized box of pixels around it. The decision can be made with multiple algorithms, this project used (cv2.ADAPTIVE_THRESH_GAUSSIAN_C).
The numbers are then thickened to compensate for any pixel values that belong to a number that turned black, thickening could reconnect that part of the number.
Then contours were found within the image to start isolating individual numbers, each contour was bound within a box and cropped out of the image. Then the number is scaled proportionally and placed within a margined frame to be like the training data. Finally, the weights of ink in the processed image are calculated to center the number in the middle based on weight and not size, like the training data. 
All the processed images of numbers are stored in an array to be sent to the model for prediction.

Prediction:
The model receives the individual number images in bulk and makes the predictions, returning the predicted vector, each prediction in the vector corresponds to a bounding box. The prediction is placed over the bounding box for each number in the original image and the final image with the bounding boxes and predictions is shown.
