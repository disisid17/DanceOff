import pygame
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import imutils
import random
import time




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
last_call_time = None





def retpose(extr=False): 
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    liine = False
    # resize the frame for portrait video

    frame = cv2.resize(frame, (int(1920 * swi / 1200), int(1080 * swi / 1200)))
    # print(frame)

    # convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process the frame for pose detection
    pose_results = pose.process(frame_rgb)
    # print(pose_results.pose_landmarks)
    if pose_results.pose_landmarks:
        # Extract landmarks
        landmarks = pose_results.pose_landmarks.landmark
        # print(landmarks)
        # landmarks = landmarks[11:16].extend(landmarks[23:28])

        # Display landmark coordinates

    landm = pose_results.pose_landmarks
    # print(landm)
    # draw skeleton on the frame
    mp_drawing.draw_landmarks(frame, landm, mp_pose.POSE_CONNECTIONS)
    # display the frame

    heigh = frame.shape[0]
    widt = frame.shape[1]
    frame = frame[0:heigh, max(0, int(widt / 2 - (heigh * 1.5) / 2)):min(widt, int(widt / 2 + (heigh * 1.5) / 2))]
    frame = cv2.resize(frame, (int(frame.shape[1] * 2 / 4), int(frame.shape[0] * 2 / 4)))
    if not extr:
        return pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "BGR")
    else:
        return pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "BGR"), frame.shape
    # Initialize Pygame



def retpos(lev, ext=False):
    model_name = f'./mediapipe-ymca/models/level{lev}_pose_model'

    suppress_landmarks = False
    add_counters = False

    with open(f'{model_name}.pkl', 'rb') as f:
        model = joblib.load(f)

    # cap = cv2.VideoCapture(0)
    # Initiate holistic model
    _, fram = cap.read()
    fram = cv2.flip(fram, 1)
    DEFAULT_IMAGE_WIDTH = fram.shape[0]
    last_detected_pose = None
    number_of_new_pose_detections = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        if cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            frame = imutils.resize(frame, width=DEFAULT_IMAGE_WIDTH)

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract Pose landmarks
                landmarks = results.pose_landmarks.landmark
                arm_landmarks = []
                pose_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.LEFT_ELBOW.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.LEFT_WRIST.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.RIGHT_WRIST.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                row = np.around(arm_landmarks, decimals=9).tolist()

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                # print(body_language_class)

                try:
                    if not ext:
                        return retpose(ext), body_language_class
                    else:
                        framt, shap = retpose(ext)
                        return framt, body_language_class, shap

                except:
                    if not ext:
                        return retpose(ext), "None"
                    else:
                        framt, shap = retpose(ext)
                        return framt, "None", shap
            except Exception as exc:
                # print(f"{exc}")
                if not ext:
                    return retpose(ext), "None"
                else:
                    framt, shap = retpose(ext)
                    return framt, "None", shap
                pass

 

pygame.init()

def last():
    """
 Returns the time since the last call to this function.

 If this is the first time the function is called, it returns 0.

 Returns:
   float: The time in seconds since the last call to this function.
 """
    global last_call_time
    now = time.time()
    if last_call_time is None:
        last_call_time = now
        return 0
    else:
        elapsed_time = now - last_call_time
        last_call_time = now
        return elapsed_time


def since():
    """
 Returns the timestamp of the last call to the "last" function.

 This function does not reset the timer like the "last" function.

 Returns:
   float: The timestamp of the last call to the "last" function.
 """
    global last_call_time
    return time.time() - last_call_time



# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
HOVER = (125, 125, 125)
YELLOW = (255, 255, 0)

# Set up the window
swi = 1500
screen = pygame.display.set_mode((swi, swi * 3 / 5))
pygame.display.set_caption("Dance Game")


# Font set

def multi(string: str, font, rect, fontColour, BGColour, justification=0):
    """Returns a surface containing the passed text string, reformatted
        to fit within the given rect, word-wrapping as necessary. The text
        will be anti-aliased.

        Parameters
        ----------
        string - the text you wish to render. \n begins a new line.
        font - a Font object
        rect - a rect style giving the size of the surface requested.
        fontColour - a three-byte tuple of the rgb value of the
                text color. ex (0, 0, 0) = BLACK
        BGColour - a three-byte tuple of the rgb value of the surface.
        justification - 0 (default) left-justified
                    1 horizontally centered
                    2 right-justified

        Returns
        -------
        Success - a surface object with the text rendered onto it.
        Failure - raises a TextRectException if the text won't fit onto the surface.
        """

    requestedLines = string.splitlines()
    # Create a series of lines that will fit on the provided
    # rectangle.

    # Let's try to write the text out on the surface.
    surface = pygame.Surface(rect.size)
    surface.fill(BGColour)
    fromt = 0
    for text in requestedLines:
        temp = font.render(text, 1, fontColour)
        surface.blit(temp, (rect.width / 2 - temp.get_width() / 2, fromt))
        fromt += font.size(text)[1]
    return surface


# Define button class
class Button:
    def __init__(self, x, y, width, height, color, text='', text_color=BLACK, fonts=36, oc=None, round=True):
        self.scale = swi / 800
        x = x * self.scale
        y = y * self.scale * 4 / 5
        self.x = x
        self.y = y
        self.width = width * self.scale
        self.height = height * self.scale
        fonts = int(self.scale * fonts)
        # self.font = pygame.font.Font('./mediapipe-ymca/ComicSansMS3.ttf', int(fonts/1.25))
        self.font = pygame.font.Font("./mediapipe-ymca/ComicSansMS3.ttf", int(fonts/1.25))
        self.rects = pygame.Rect(x - self.width * 0.025, y - self.width * 0.025, self.width * 1.05,
                                 self.height + self.width * 0.05)
        self.rect = pygame.Rect(x, y, self.width, self.height)

        self.color = color
        self.oc = oc
        self.text = text
        self.text_color = text_color
        self.make = True
        self.calls = 0
        if not round:
            self.scale = 0

    def draw(self, surface, ml=False):
        if self.calls == 0:
            print(self.text)
            self.calls += 1
        if self.make:
            if self.oc:
                pygame.draw.rect(surface, self.oc, self.rects, 0, border_radius=int(self.scale * 10))
            pygame.draw.rect(surface, self.color, self.rect, 0, border_radius=int(self.scale * 10))

            if self.text != '':
                if not ml:
                    text_surface = self.font.render(self.text, True, self.text_color)
                    text_rect = text_surface.get_rect()
                    text_rect.center = self.rect.center
                    surface.blit(text_surface, text_rect)
                if ml:
                    text_surface = multi(self.text, self.font, self.rect, self.text_color, self.color)
                    text_rect = text_surface.get_rect()
                    text_rect.center = self.rect.center
                    surface.blit(text_surface, (self.x, self.y))

    def texty(self, surface, tex="", coord=(0, 0)):
        if self.make:
            pygame.draw.rect(surface, WHITE, self.rect, 0, border_radius=int(self.scale * 10))
            pygame.draw.rect(surface, WHITE, self.rect, 0, border_radius=int(self.scale * 10))
            if self.text != '':
                text_surface = self.font.render(tex, True, RED)
                text_rect = text_surface.get_rect()
                text_rect.center = self.rect.center
                surface.blit(text_surface, coord)

    def rem(self):
        self.make = False

    def is_hover(self, pos):
        return self.rect.collidepoint(pos)


# Function to create buttons
def create_buttons():
    easy_button = Button(100, 200, 200, 50, GREEN, 'Easy', oc=GREEN)
    medium_button = Button(100, 300, 200, 50, BLUE, 'Medium', oc=YELLOW)
    hard_button = Button(100, 400, 200, 50, RED, 'Hard', oc=RED)
    return [easy_button, medium_button, hard_button]


def level_select_screen():
    screen.fill(BLUE)
    l1 = Button(300, 150, 200, 50, GREEN, "YMCA", oc=GREEN)
    l2 = Button(300, 250, 200, 50, BLACK, "Level 2", oc=YELLOW)
    hard_button = Button(300, 350, 200, 50, RED, 'Level 3', oc=RED)
    return [l1, l2, hard_button]
    # pygame.display.flip()


def exitb():
    screen.fill(HOVER)
    return Button(300, 250, 400, 100, WHITE, "Restart?", oc=RED)


def main_menu():
    running = True
    dif = 0
    lev = 0
    buttons = create_buttons()
    levs = level_select_screen()
    welc = Button(300, 50, 200, 100, WHITE, "Welcome to Dance-off!", RED, fonts=48)
    difsel = False
    levsel = False
    while not levsel:

        if not difsel:
            screen.fill(WHITE)
        else:
            screen.fill(BLUE)
        mouse_pos = pygame.mouse.get_pos()
        if not difsel:

            welc.draw(screen)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for button in buttons:
                        if button.is_hover(mouse_pos) and button.make:
                            if True:
                                if button.text == 'Easy':
                                    dif = 3
                                elif button.text == 'Medium':
                                    dif = 2
                                else:
                                    dif = 1
                                button.rem()
                            print(f"Clicked {button.text} difficulty")
                            difsel = True
            for button in buttons:
                if button.is_hover(mouse_pos):
                    button.color = HOVER
                else:
                    button.color = WHITE

                button.draw(screen)

        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for button in levs:
                        if button.is_hover(mouse_pos) and button.make:
                            if True:
                                if button.text == 'YMCA':
                                    lev = 1
                                elif button.text == 'Level 2':
                                    lev = 2
                                else:
                                    lev = 3
                                button.rem()
                            print(f"Clicked {button.text}")
                            levsel = True
            for button in levs:
                if button.is_hover(mouse_pos):
                    button.color = HOVER
                else:
                    button.color = WHITE
                button.draw(screen)

        pygame.display.flip()
    return lev, dif


def game(lev, dif):
    timeTo = 2 + dif * 2
    scom = 0.95 + (3 - dif) * 0.05
    ex = False

    order = leva(lev)
    last()
    count = 0
    ci = 0
    buts = []
    col = None
    match dif:
        case 3:
            col = GREEN
        case 2:
            col = YELLOW
        case 1:
            col = RED
    while not ex:
        mouse_pos = pygame.mouse.get_pos()

        if count == 0:
            last()
            count += 1

        # scre,pos = retpos()
        # screen.blit(scre,(200,50))
        screen.fill(WHITE)

        if (ci == 0):
            buts = shot(order)
            buttonyy = Button(300, 250, 200, 50, WHITE, "Continue?", BLACK, 36, col)

            ci += 1
        for i in buts:
            i.draw(screen)
        if buttonyy.is_hover(mouse_pos):
            buttonyy.color = HOVER
        else:
            buttonyy.color = WHITE
        buttonyy.draw(screen)
        pygame.display.flip()
        if since() > timeTo:
            # ex = True
            pass
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ex = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if buttonyy.is_hover(mouse_pos) and buttonyy.make:
                    ex = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                ex = True

    ex = False

    while not ex:
        screen.fill(WHITE)
        scre, pos = retpos(lev)
        last()
        times = []
        comp = 0
        more = []
        back = Button(15, 15, swi / 1.95, swi * 4 / 10, WHITE, '', RED, fonts=24, oc=RED, round=False)
        for i in range(len(order)):
            more.append("?")

        for act in order:
            top = "Completed Moves:"
            for i in more:
                top += ("\n" + i)
            we = Button(560, 15, 200, 400, WHITE, top, RED, fonts=24, round=False)
            while pos != act:

                screen.fill(WHITE)
                back.draw(screen)
                scre, pos, shap = retpos(lev, True)
                high = Button(10, 10, shap[1] / 1.90, shap[0] / 1.915, WHITE, '', oc=GREEN, fonts=24, round=False)
                high.draw(screen)
                screen.blit(scre, (10, 10))

                wel = Button(20, 480, 600, 100, WHITE, '', RED, fonts=24, oc=RED, round=False)
                welar = Button(140, 450, 200, 100, WHITE, (
                            "Current Move: " + str(pos) + "\n" + "\nPossible Score: " + str(
                        max(0, round(8 - since(), 3)))), RED, fonts=24, round=False)

                we.draw(screen, True)

                wel.draw(screen)
                welar.draw(screen, True)
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        ex = True
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        ex = True
            more[comp] = act
            comp += 1
            pygame.display.flip()
            itim = last()
            if (itim < timeTo):
                times.append((act, round(8 - itim, 3), round(itim, 3)))
            else:
                times.append((act, 0, round(itim, 3)))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    ex = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    ex = True

        pygame.display.flip()
        for ac in times:
            print(ac, end=" ")
        ex = True
        # shot(order)
        toret = score(scom, times)

        return toret

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ex = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                ex = True


def score(mult, times):
    sut = 0
    scale = 8 * len(times)
    for i in times:
        sut += i[1]
    scale = sut / scale

    sut = round((scale * 50000), 3) * mult
    print(sut)
    return sut


def leva(lev):
    with open(f'./mediapipe-ymca/models/level{lev}_pose_model_classes.txt', 'r') as f:
        ret = f.read()
    ret = ret[1:len(ret) - 1]
    ret = ret.replace("\'", '')
    ret = ret.split()
    print(ret)
    states = ret
    random.shuffle(states)
    return states


def shot(orda):
    ret = []
    ret.append(Button(400, 50, 20, 10, WHITE, str(orda), RED, fonts=48))

    for i, act in enumerate(orda):
        ret.append(Button(30 + i * 60, 100 + i * 00, 20, 10, WHITE, str(act), RED, fonts=36))
    return ret


def redo():
    screen.fill(WHITE)
    sel = False
    exi = Button(200, 250, 400, 100, WHITE, "Restart?", oc=RED)
    while not sel:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if exi.is_hover(mouse_pos) and exi.make:
                    exi.rem()
                    print("Restarting")
                    sel = True

        if exi.is_hover(mouse_pos):
            exi.color = HOVER
        else:
            exi.color = WHITE

        exi.draw(screen)
        pygame.display.flip()


##
# Run the main menu
while True:
    lev, dif = main_menu()
    print(dif)
    print(lev)
    psco = game(lev, dif)
    redo()
