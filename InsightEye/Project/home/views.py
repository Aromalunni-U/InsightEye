from django.shortcuts import render,redirect
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import time
import google.generativeai as genai
import markdown2  
from  pyttsx3 import init


def to_markdown(text):
    return markdown2.markdown(text)



api_key = "AIzaSyCDAgWInZtJI1Y6JV44Q_F4W2ZFLxBBnQY"
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=1)


def cal_ear(eye_landmarks):
    
    hor_line = dist.euclidean(eye_landmarks[0],eye_landmarks[1])
    ver_line1 = dist.euclidean(eye_landmarks[2],eye_landmarks[3])
    ver_line2 = dist.euclidean(eye_landmarks[4],eye_landmarks[5])

    return (ver_line1 + ver_line2) / (2 * hor_line)



def index(request):
    return render(request,"index.html")

def code(request):
    return render(request,"code.html")

def test(request): 
    return render(request,"test.html")



def camera(request):

    eye_closed_frame = 0
    count = 0
    time_started = False

    #Gaze stability variables
    prev_left_pupil = None
    prev_right_pupil = None
    unstable_frames = 0

    #Saccadic speed variables
    prev_time = None
    saccadic_speeds = []

    #Eye landmarks
    LEFT_EYE = [33, 133, 160, 159, 158, 153]
    RIGHT_EYE = [362, 263, 387, 386, 385, 380]

    Video = cv2.VideoCapture(0)

    while True:
        success, img = Video.read()
        if not success:
            break

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            if not time_started:
                start_time = time.time()
                time_started = True

            for fmark in results.multi_face_landmarks:
                h, w, z = img.shape

                # ----------------------Pupil coloring-------------------------------------

                left_pupil = fmark.landmark[468]
                right_pupil = fmark.landmark[473]
                left_pupil_coord = (int(left_pupil.x * w), int(left_pupil.y * h))
                right_pupil_coord = (int(right_pupil.x * w), int(right_pupil.y * h))
                cv2.circle(img, left_pupil_coord, 4, (0, 0, 255), -1)
                cv2.circle(img, right_pupil_coord, 4, (0, 0, 255), -1)

                #--------------Gaze stability and Saccadic Speed analysis-------------------

                if prev_left_pupil and prev_right_pupil:
                    left_movement = dist.euclidean(prev_left_pupil, left_pupil_coord)
                    right_movement = dist.euclidean(prev_right_pupil, right_pupil_coord)

                    if left_movement > 10 or right_movement > 10:
                        unstable_frames += 1

                    #-----------------------Calculating Saccadic Speed-----------------------

                    current_time = time.time()
                    if prev_time is not None:
                        time_elapsed = current_time - prev_time
                        avg_movement = (left_movement + right_movement) / 2
                        saccadic_speed = avg_movement / time_elapsed
                        saccadic_speeds.append(saccadic_speed)

                    prev_time = current_time

                prev_left_pupil = left_pupil_coord
                prev_right_pupil = right_pupil_coord

                #----------------------------------Blink detection-------------------------------------------

                left_eye = [(int(fmark.landmark[i].x * w), int(fmark.landmark[i].y * h)) for i in LEFT_EYE]
                right_eye = [(int(fmark.landmark[i].x * w), int(fmark.landmark[i].y * h)) for i in RIGHT_EYE]

                if len(left_eye) == 6 and len(right_eye) == 6:
                    left_eye_ear = cal_ear(left_eye)
                    right_eye_ear = cal_ear(right_eye)
                    ear = (left_eye_ear + right_eye_ear) / 2

                    if ear < 0.2:
                        eye_closed_frame += 1
                    else:
                        if eye_closed_frame > 1:
                            count += 1
                        eye_closed_frame = 0

                    cv2.putText(img, f"Blinks: {count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Video", img)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break

    Video.release()
    cv2.destroyAllWindows()

    if time_started:
        total_time = int(time.time() - start_time)

    gaze_stability = "Low" if unstable_frames > (total_time/2) else "High"  
    avg_saccadic_speed = sum(saccadic_speeds) / len(saccadic_speeds) if saccadic_speeds else 0

    print(f"Average Saccadic Speed: {avg_saccadic_speed:.2f} units/second")



    eye_tracking_data = f"Blink rate: {count} blinks/{total_time} second, Gaze stability: {gaze_stability}, Average Saccadic Speed: {avg_saccadic_speed:.2f} units/second."
    response = model.generate_content(f"Analyze the following data and provide mental health insights:\n{eye_tracking_data}")
    text = to_markdown(response.text)

    return render(request,"test.html",{"text":text})



