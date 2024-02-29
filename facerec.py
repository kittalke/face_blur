import cv2
import face_recognition
import os

def cut_and_save_faces(video_path, output_folder):
    video_capture = cv2.VideoCapture(video_path)

    ret, frame = video_capture.read()

    face_locations = face_recognition.face_locations(frame)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        face_image = frame[top:bottom, left:right]

        face_filename = f"{output_folder}/face_{i+1}.jpg"
        cv2.imwrite(face_filename, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        print(f"Twarz {i+1} zapisana jako {face_filename}")

    video_capture.release()

def choose_faces(face_folder):
    face_files = [f"{face_folder}/{file}" for file in sorted(os.listdir(face_folder))]

    window_name = "Wybierz twarze do zamazania"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    chosen_faces = []
    for i, face_file in enumerate(face_files):
        face_image = cv2.imread(face_file)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        cv2.imshow(window_name, face_image)

        key = cv2.waitKey(0)

        if key == ord('y'):
            chosen_faces.append(i)

    cv2.destroyWindow(window_name)

    return chosen_faces

def recognize_faces(video_path, face_folder, chosen_faces, output_path):
    video_capture = cv2.VideoCapture(video_path)
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    face_files = [f"{face_folder}/{file}" for file in sorted(os.listdir(face_folder))]

    known_face_encodings = []
    known_face_names = []

    for i in chosen_faces:
        face_file = face_files[i]
        face_image = face_recognition.load_image_file(face_file)
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(os.path.basename(face_file))[0])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
                
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Nieznany"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                frame[top:bottom, left:right] = cv2.GaussianBlur(frame[top:bottom, left:right], (99, 99), 30)

        out.write(frame)
        frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("koniec")
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video3.mp4"

    output_folder = "/home/rita/Documents/Inynmierka/faces"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cut_and_save_faces(video_path, output_folder)

    chosen_faces = choose_faces(output_folder)

    recognize_faces(video_path, output_folder, chosen_faces, "output.mp4")
