from ultralytics import YOLO
import cv2 as cv
import math

def run():
    model = YOLO(r"Path_to_model") 
    # If you want to use a video just pass the vidoe path.
    # Use 0 for using your webcam.  
    cap = cv.VideoCapture(0)
    # Every parameter for writing confidence.
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    while True:
        # Getting frame.
        _,frame =cap.read()
        # Flipping frame.
        frame = cv.flip(frame,1)
        # Predicting faces in frame.
        resulte = model.predict(frame,stream=True)
        for r in resulte:
            # Getting boxes from predictions.
            boxes = r.boxes
            for box in boxes:
                # Bounding boxs
                x1, y1, x2, y2 = box.xyxy[0]
                x,y,w,h = box.xywh[0]
                x,y,w,h= int(x), int(y), int(w), int(h)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
               
                # Put box in frame.
                roi = frame[y1:y1+h,x1:x1+w]
                roi = cv.GaussianBlur(roi,(23,23),30)
                
                # Confidence.
                confidence = math.ceil((box.conf[0]*100))/100
                frame[y1:y1+roi.shape[0], x1:x1+roi.shape[1]] = roi
                # Writing confidence in frame.
                cv.putText(frame, f"face | {confidence}", [x1, y1], font, fontScale, color, thickness)

        # Showing final frame.
        cv.imshow('Webcam', frame)
        # Press 'q' to close program
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    run()
