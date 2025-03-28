import cv2
from deepface import DeepFace
import webbrowser
import threading
import time

# Predefined YouTube Playlists for Different Emotions 
emotion_playlists = {
    "happy": "https://youtu.be/hN4f9POjgEk?si=S3hR3cjsB8nPFB8e",
    "sad": "https://youtube.com/playlist?list=PL115iZFgSUHaEbv9Why0FV7jvAN4qREdJ&si=ycGxad1cNpvSIvhh",
    "angry": "https://youtube.com/playlist?list=PLvnIhUj6CayCfcOmJB-jdvu29pw-EpZEy&si=0FUroViFKFLiy3fE",
    "neutral": "https://youtu.be/pIvf9bOPXIw?si=9Ii0Q9iaw6FNbz85",
    "fear": "https://youtube.com/playlist?list=RDD6MOuX980gc&playnext=1&si=yzcCUb6vp0HU8-uK",
    "surprise": "https://www.youtube.com/watch?v=s1mUujXO_I0&list=PLjIydaclej7C1NDDi_vMmlKy7UeIZ_e9Z",
}

def capture_frame():
    """Capture a frame from the webcam with improved stability and resolution."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No accessible webcam found.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

    time.sleep(2)  # Allow camera to adjust to lighting conditions
    print("Capturing image for emotion detection...")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Unable to capture frame.")
        return None
    return frame

def analyze_emotion(frame, result_container):
    """Analyze the emotion in the frame using DeepFace."""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Using different detector backends to improve accuracy
        detector_backends = ['mtcnn', 'ssd', 'mediapipe']

        for backend in detector_backends:
            print(f"Trying backend: {backend}")
            try:
                result = DeepFace.analyze(
                    rgb_frame, 
                    actions=['emotion'], 
                    detector_backend=backend, 
                    enforce_detection=False
                )

                if isinstance(result, list):
                    result = result[0]  # Handle list response

                result_container['emotion'] = result.get("dominant_emotion", None)
                result_container['region'] = result.get("region", {})
                result_container['details'] = result.get("emotion", {})  # Store all confidence scores

                if result_container['emotion'] != "neutral":
                    return  # Stop if a strong emotion is found
            
            except Exception as e:
                print(f"Error with backend {backend}: {e}")

    except Exception as e:
        print(f"Emotion detection error: {e}")
        result_container['emotion'] = None
        result_container['region'] = {}

def display_result(frame, emotion, region, details):
    """Display the detected emotion on the image and open the recommended playlist."""
    if emotion:
        print(f"\nDetected Emotion: {emotion}")
        print("Confidence Scores:", details)

        if region and "x" in region:
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text_x = max(x, 10)
            text_y = max(y - 10, 30)
            cv2.putText(frame, f"Emotion: {emotion}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Emotion: {emotion}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

        playlist_url = emotion_playlists.get(emotion.lower())
        if playlist_url:
            print(f"Opening Playlist: {playlist_url}")
            webbrowser.open(playlist_url)
        else:
            print("No predefined playlist found for this emotion.")
    else:
        print("No dominant emotion detected.")

def detect_emotion_and_recommend_playlist():
    """Main function to detect emotion and recommend a playlist."""
    frame = capture_frame()
    if frame is None:
        return

    result_container = {"emotion": None, "region": {}, "details": {}}
    
    thread = threading.Thread(target=analyze_emotion, args=(frame, result_container))
    thread.start()
    thread.join()

    display_result(frame, result_container['emotion'], result_container['region'], result_container['details'])

if __name__ == "__main__":
    detect_emotion_and_recommend_playlist()
