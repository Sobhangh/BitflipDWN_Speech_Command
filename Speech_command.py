import kagglehub
import os
import random
import numpy as np

# Download latest version
path = kagglehub.dataset_download("yashdogra/speech-commands")

print("Path to dataset files:", path)

# Get all folders in the dataset
all_items = os.listdir(path)
folders = [item for item in all_items if os.path.isdir(os.path.join(path, item))]
print(f"\nFolders in the dataset: {folders}")

if folders:
    # Select a random folder
    random_folder = random.choice(folders)
    folder_path = os.path.join(path, random_folder)
    print(f"\nRandom folder selected: {random_folder}")
    
    # Get files in the random folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    #print(f"\nFiles in the random folder: {files}")
    
    if files:
        # Select a random file
        random_file = random.choice(files)
        file_path = os.path.join(folder_path, random_file)
        print(f"Random file selected: {random_file}")
        
        # Read and play the audio file
        import wave
        try:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                print(f"\nAudio file properties:")
                print(f"Channels: {channels}, Sample Width: {sample_width}, Frame Rate: {frame_rate} Hz")
                print(f"Duration: {len(frames) / (frame_rate * channels * sample_width):.2f} seconds")
                

                audio_array = np.frombuffer(frames, dtype=np.int16)  # for 16-bit (sample_width=2) audio
                print(len(frames))
                print(audio_array.max(), audio_array.min())  # now these are integers like [-32, 0, 128, ...]
        except Exception as e:
            print(f"Error reading audio file: {e}")

