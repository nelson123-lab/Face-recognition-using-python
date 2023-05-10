import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load face data
face_data = np.load('face_data.npy')
labels = np.load('labels.npy')

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(face_data, labels)

# Initialize OpenCV face detection cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV face detection cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Extract face features using OpenCV face recognition
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100)).flatten()

        # Predict label using KNN classifier
        label = knn.predict([face])[0]

        # Draw label on frame
        cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw bounding box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
