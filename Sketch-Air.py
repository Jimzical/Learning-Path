import cv2
import mediapipe as mp
from time import sleep
import numpy as np
from math import sqrt

def initializeCode(hands, cap):
    """
    -------------------------------------------------------------
    Initializes the code
        - Takes the camera input
        - Flips the image
        - Processes the image (for results)
    -------------------------------------------------------------
    ### Parameters:
        hands: Hand object [mediapipe object]
        cap: Camera object [cv2 object]

    ### Returns:
        image: Image from the camera [numpy array]
        results: Results from the image [mediapipe object] {For Hand Detection}
    """
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
        print("Ignoring empty camera frame.")
        # TODO: Make it Raise an error Later
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = hands.process(image)
    return image, results


def endCode(cap, hands, debug=False):
    """
    -------------------------------------------------------------
    Ends the code
        - Releases the camera
        - Destroys all the windows
    -------------------------------------------------------------
    ### Parameters:
        cap: Camera object [cv2 object]
        debug: Debug mode [bool] (Default = False) {If True, It will Wait for a Key Press to take a new Frame}

    ### Returns:
        True: If the code is ended [bool]
        False: If the code is not ended [bool]
    """
    if debug:
        WaitVal = 0
    else:
        WaitVal = 50

    if cv2.waitKey(WaitVal) == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

        return True
    else:
        return False


def markHands(
    image, results, point, mp_drawing, mp_drawing_styles, mp_hands
):
    """
    -------------------------------------------------------------
    Marks the hands on the Image and Prints the Coordinates
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        results: Results from the image [mediapipe object] {For Hand Detection}
        point: Points on the hand [dict]
        mp_drawing: Drawing object [mediapipe object]
        mp_drawing_styles: Drawing Styles object [mediapipe object]
        mp_hands: Hands object [mediapipe object]

    ### Returns:
        None
    """
    # making the marks on the hands
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


def showImage(image, changeSize=False):
    """
    -------------------------------------------------------------
    Shows the Resized Image
    -------------------------------------------------------------
    ## Parameters:
        image: Image from the camera [numpy array]
        changeSize: Change the size of the image [bool] (Default = False) {If True, It will Resize the Image}
    ### Returns:
        None
    """
    if changeSize:
        cv2.namedWindow("MediaPipe Hands", cv2.WINDOW_NORMAL)
    cv2.imshow("MediaPipe Hands", image)
    return


def draw(image,coordinate_list,color = (0, 0, 255), radius = 4 ):
    '''
    -------------------------------------------------------------
    Draws a blob on the finger
    -------------------------------------------------------------
     ### Parameters:
        image: Image from the camera [numpy array]
        coordinate_list: List of coordinates to draw [list(tuple(int x, int y))]
        color: Color of the blob [tuple(int B, int G, int R)] (Default = (0, 0, 255))
        radius: Radius of the blob [int] (Default = 4)

    ### Returns:
        None
    '''
    for point in coordinate_list:
        cv2.circle(image, (point[0], point[1]), radius, color, -1)
    
    return
def AddBlobToFinger(image, finger, color = (0, 0, 255), radius = 25 , thickness = 2):
    '''
    -------------------------------------------------------------
    Draws a blob on the finger
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        finger: Finger to check [HandLandmark]
        color: Color of the blob [tuple(int B, int G, int R)] (Default = (0, 0, 255))
        radius: Radius of the blob [int] (Default = 25)

    ### Returns:
        None
    '''

    radius = int(radius * abs(finger.z) ** 0.2)
    cv2.circle(image, (int(finger.x * image.shape[1]), int(finger.y * image.shape[0])), radius, color,thickness)
 

def CheckContactBtwFingers(image, finger1, finger2):
    '''
    -------------------------------------------------------------
    Checks if two fingers are in contact
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        finger1: First finger to check [HandLandmark]
        finger2: Second finger to check [HandLandmark]

    ### Returns:
        True: If the fingers are in contact [bool]
        False: If the fingers are not in contact [bool]
    '''
    dist = sqrt((finger1.x - finger2.x) ** 2 + (finger1.y - finger2.y) ** 2 )
    # print("dist: {}".format(dist))
    if dist < 0.1:
        return True
    else:
        return False

def colors():
    '''
    -------------------------------------------------------------
    Returns a dictionary of colors
    -------------------------------------------------------------
    ### Parameters:
        None
    ### Returns:
        colors: Dictionary of colors [dict]
    '''
    return {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "orange": (0, 165, 255),
        "purple": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

def  draw_triange(image, point1, point2, point3, color = (0, 0, 255)):
    '''
    -------------------------------------------------------------
    Draws a triangle on the image
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        point1: First point of the triangle [tuple(int x, int y)]
        point2: Second point of the triangle [tuple(int x, int y)]
        point3: Third point of the triangle [tuple(int x, int y)]
        color: Color of the triangle [tuple(int B, int G, int R)] (Default = (0, 0, 255))

    ### Returns:
        None
    '''
    cv2.putText(image,"Size", (point1[0] - 50, point1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    pts = np.array([point1, point2, point3])
    cv2.fillPoly(image, [pts], color)
    return

def CheckSize(fingerX, fingerY):
    '''
    -------------------------------------------------------------
    Checks the location of the finger and returns the size of the pencil based on the location
    -------------------------------------------------------------
    ### Parameters:
        fingerX: X coordinate of the finger [float]
        fingerY: Y coordinate of the finger [float]

    ### Returns:
        size: Size of Pencil [int]
    '''
    upper = [590, 620]
    lower = [50 , 400]

    if fingerX > upper[0] and fingerX < upper[1] and fingerY > lower[0] and fingerY < lower[1]:
        size = fingerY//10 - 8
        # print("size: {}, type: {}".format(size, type(size)))
        return int(size)

def draw_rect(image, point1, point2, color = (0, 0, 255)):
    '''
    -------------------------------------------------------------
    Draws a rectangle on the image
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]  
        point1: First point of the rectangle [tuple(int x, int y)]
        point2: Second point of the rectangle [tuple(int x, int y)]
        color: Color of the rectangle [tuple(int B, int G, int R)] (Default = (0, 0, 255))

    ### Returns:
        None
    '''
    cv2.rectangle(image, point1, point2, color, -1)


def CheckColor(image,fingerX, fingerY, colorlist):
    '''
    -------------------------------------------------------------
    Checks the color of the finger
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        fingerX: X coordinate of the finger [float]
        fingerY: Y coordinate of the finger [float]
        colorlist: Dictionary of colors [dict]

    ### Returns:
        chosen_color: Color of the finger [tuple(int B, int G, int R)]
    '''

    if fingerY > 0 and fingerY < 50:
        if fingerX > 0 and fingerX < 50:
            chosen_color = colorlist["red"]
        elif fingerX > 50 and fingerX < 100:
            chosen_color = colorlist["green"]
        elif fingerX > 100 and fingerX < 150:
            chosen_color = colorlist["blue"]
        elif fingerX > 150 and fingerX < 200:
            chosen_color = colorlist["yellow"]
        elif fingerX > 200 and fingerX < 250:
            chosen_color = colorlist["orange"]

        return chosen_color

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands


    colorlist = colors()

    chosen_color = colorlist["red"]

    # For webcam input:
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.9, min_tracking_confidence=0.9
    )

    coords = [ [0 , 0 ] ]
    image, results = initializeCode(hands, cap)
    

    horizontal_fix = image.shape[1]
    vertical_fix = image.shape[0]
    PencilSize = 4
    while cap.isOpened():
        image, results = initializeCode(hands, cap)

        draw_triange(image, (595, 50), (580, 400), (620, 400), color = chosen_color)
        draw_rect(image, (0 ,  0), (50 , 50), color = colorlist["red"])
        draw_rect(image, (50,  0), (100, 50), color = colorlist["green"])
        draw_rect(image, (100, 0), (150, 50), color = colorlist["blue"])
        draw_rect(image, (150, 0), (200, 50), color = colorlist["yellow"])
        draw_rect(image, (200, 0), (250, 50), color = colorlist["orange"])

        if coords != [ [0 , 0 ] ]:
            draw(image,coords,color = chosen_color , radius = int (PencilSize))

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1:
            cv2.putText(image, "Hand Detected", (10, 440), cv2.FONT_HERSHEY_PLAIN, 2, colorlist["green"], 2)
            
            hand = {
                "Index Tip": results.multi_hand_landmarks[0].landmark[8],
                "Thumb Tip": results.multi_hand_landmarks[0].landmark[4],
            }

            sizeCheck = CheckSize(hand["Index Tip"].x * horizontal_fix, hand["Index Tip"].y * vertical_fix)
            if type(sizeCheck) == int:
                PencilSize = sizeCheck
            colorCheck = CheckColor(image, hand["Index Tip"].x * horizontal_fix, hand["Index Tip"].y * vertical_fix, colorlist= colorlist)
            if type(colorCheck) == tuple:
                chosen_color = colorCheck
            
            # markHands(image, results, hand, mp_drawing, mp_drawing_styles, mp_hands)

            AddBlobToFinger(image, hand["Index Tip"], color = colorlist["red"])
            AddBlobToFinger(image, hand["Thumb Tip"], color = colorlist["orange"])
            
            if CheckContactBtwFingers(image, hand["Index Tip"], hand["Thumb Tip"]):
                cv2.putText(image, "ON", (10, 470), cv2.FONT_HERSHEY_PLAIN, 2, colorlist["green"], 2)
                coords.append([int(hand["Index Tip"].x * horizontal_fix), int(hand["Index Tip"].y * vertical_fix)])
            else:
                cv2.putText(image, "OFF", (10, 470), cv2.FONT_HERSHEY_PLAIN, 2, colorlist["red"], 2)
        else:
            cv2.putText(image, "Hand Not Detected", (10, 440), cv2.FONT_HERSHEY_PLAIN, 2, colorlist["red"], 2)
        showImage(image)
        endCode(cap, hands)


if __name__ == "__main__":
    main()
