import tkinter as tk
from tkinter import filedialog
import subprocess
from PIL import ImageChops, Image
import cv2
import numpy as np

class ImageAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Analysis")
        self.root.geometry("500x400")
        self.image_file_path = None

        self.analysis_text = tk.Text(self.root, wrap=tk.WORD, width=60, height=20)
        self.analysis_text.pack()

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.get_image_metadata)
        self.upload_button.pack(side="top", pady=10)

        self.generate_report_button = tk.Button(self.root, text="Generate a Report", command=self.generate_report)
        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.destroy)
        self.ela_button = tk.Button(self.root, text="Perform ELA Analysis", command=self.perform_ela_analysis)

    def get_image_metadata(self):
        self.image_file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg")])

        if self.image_file_path:
            try:
                metadata_output = subprocess.check_output(["exiftool", self.image_file_path])
                self.display_metadata(metadata_output)

                self.upload_button.pack_forget()

                self.ela_button.pack(side="top", pady=10)
                self.generate_report_button.pack(side="top", pady=10)
                self.exit_button.pack(side="top", pady=10)
            except subprocess.CalledProcessError:
                error_message = tk.Label(self.root, text="Error extracting metadata. Please make sure exiftool is installed.")
                error_message.pack()

    def display_metadata(self, metadata):
        self.analysis_text.delete("1.0", tk.END)
        self.analysis_text.insert(tk.END, metadata.decode("utf-8"))

    def perform_ela_analysis(self):
        if self.image_file_path:
            temp = 'temp.jpg'
            scale = 10
            original = Image.open(self.image_file_path)
            original.save(temp, quality=90)
            temporary = Image.open(temp)
            diff = ImageChops.difference(original, temporary)
            d = diff.load()
            width, height = diff.size
            for x in range(width):
                for y in range(height):
                    d[x, y] = tuple(k * scale for k in d[x, y])

            diff.save("ela_result.jpg")
            rec = cv2.face_LBPHFaceRecognizer.create()
            rec.read("TrainedDataFolder/TraningData.yml")
            imggray = Image.open("ela_result.jpg").convert('L')
            gray = np.array(imggray, 'uint8')
            self.analysis_text.delete("1.0", tk.END)
            self.analysis_text.insert(tk.END, "Doing ELA analysis...\n")
            id, conf = rec.predict(gray)
            result_text = f"Result: {'REAL' if id == 2 else 'FAKE'} - Confidence: {100 - conf:.2f}%"
            self.analysis_text.insert(tk.END, result_text)

            self.ela_button.pack_forget()

    def generate_report(self):
        # Add your report generation code here
        pass

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageAnalysisApp()
    app.run()
