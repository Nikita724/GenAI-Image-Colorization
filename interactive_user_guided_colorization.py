import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

def upload_image():
    file_path = filedialog.askopenfilename()
    img = plt.imread(file_path)
    plt.imshow(img)
    plt.title("Uploaded Image")
    plt.show()

def main():
    root = tk.Tk()
    root.title("Interactive User-Guided Colorization")
    upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
    upload_btn.pack()
    root.mainloop()

if __name__ == "__main__":
    main()
