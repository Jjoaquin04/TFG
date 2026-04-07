import cv2

def save_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file")
        return
    
    # Create the output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through frames and save them
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame as PNG. Starts downloading at frame 10200 since we already had the first 10199 frames in the previous container
        if frame_count > 737:
            print(f'frame numero: { frame_count}\n')
            frame_path = os.path.join(output_folder, f'frame_{str(frame_count).zfill(6)}.png')
            cv2.imwrite(frame_path, frame)

        frame_count += 1
    
    # Release the video capture object
    cap.release()

# Example usage
VIDEO_PATH = "data/inputs/videos/2022_BCN_FinalF_1.mp4" # Path to video
output_folder = '/images/train2'  # Change this to the folder where you want to save frames
save_frames(VIDEO_PATH, output_folder)
