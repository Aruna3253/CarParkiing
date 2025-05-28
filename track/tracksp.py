import cv2
import numpy as np
import pickle

# Load video and polygon positions
cap = cv2.VideoCapture('../carPark.mp4')
with open('polygon_spots.pkl', 'rb') as f:
    polygon_spots = pickle.load(f)

# Threshold: determines if a spot is occupied
OCCUPIED_THRESHOLD = 0.2
font = cv2.FONT_HERSHEY_SIMPLEX

def count_white_pixels_inside_polygon(thresh_img, polygon):
    mask = np.zeros(thresh_img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)

    masked_img = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
    white_pixels = cv2.countNonZero(masked_img)
    area = cv2.countNonZero(mask)

    ratio = white_pixels / area if area > 0 else 0
    return ratio

while True:
    # Loop the video
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    overlay = frame.copy()

    # Processing for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)

    free_count = 0
    for idx, polygon in enumerate(polygon_spots):
        ratio = count_white_pixels_inside_polygon(thresh, polygon)

        if ratio < OCCUPIED_THRESHOLD:
            color = (0, 255, 0) 
            free_count += 1
        else:
            color = (0, 0, 255) 

        pts = np.array(polygon, np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(overlay, f"{ratio:.2f}", tuple(pts[0]), font, 0.6, color, 2)

    # Show total free count
    cv2.rectangle(overlay, (0, 0), (250, 40), (50, 50, 50), -1)
    cv2.putText(overlay, f"{free_count}/{len(polygon_spots)} Free", (10, 30),
                font, 0.8, (255, 255, 255), 2)

    cv2.imshow("Parking Spot Detection", overlay)

    # Exit on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Quit requested by user.")
        break

cap.release()
cv2.destroyAllWindows()
