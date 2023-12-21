import cv2

# for a face detection we require certain features to be detected by our camera
# we write about certain face features in the code

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    "C:/Users/Navdeep/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

# Open the webcam
video_capture = cv2.VideoCapture(0)

# Loop until the user quit
while True:

    # Capture a frame from the webcam
    ret, video_frame = video_capture.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Draw a rectangle around each detected face
    for (u, v, w, h) in faces:
        cv2.rectangle(video_frame, (u, v), (u + w, v + h), (0, 255, 0), 2)

    # Display the frame with the detected faces
    cv2.imshow("video_live", video_frame)

    # If the user presses the 'a' key, quit the loop
    if cv2.waitKey(10) == ord("a"):
        break

# Release the webcam
video_capture.release()