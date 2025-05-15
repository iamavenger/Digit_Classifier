import numpy as np
from mnist_loadcsv import load_mnistcsv
import tkinter as tk
from collections import Counter
from PIL import Image, ImageDraw

# KNN Implementation
class KNN:
    def __init__(self, k=3):
        self.k = k  

    def fit(self, train_data, train_labels):
        self.train_data = train_data.reshape(train_data.shape[0], -1)  # Flatten images
        self.train_labels = train_labels

    def predict(self, test_sample):
        test_sample = test_sample.flatten()  # Flatten the image
        distances = np.linalg.norm(self.train_data - test_sample, axis=1)
        nearest_neighbors = np.argsort(distances)[:self.k]
        neighbor_labels = self.train_labels[nearest_neighbors]
        return Counter(neighbor_labels).most_common(1)[0][0]

# Load dataset
(train_images, train_labels) = load_mnistcsv("MNIST_DATASET_CSV\mnist_train.csv")

# Train KNN
knn = KNN(k=3)
knn.fit(train_images, train_labels)

# Interactive Drawing Interface
class DigitDrawer:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.canvas_size = 280
        self.cell_size = self.canvas_size // 28  # Each cell corresponds to one MNIST pixel
        self.prediction_label = tk.StringVar()
        
        # Create canvas for drawing
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons
        self.button_predict = tk.Button(root, text="Predict", command=self.predict_digit)
        self.button_predict.pack()
        self.button_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        
        # Label to display prediction
        self.result_label = tk.Label(root, textvariable=self.prediction_label, font=("Arial", 20))
        self.result_label.pack()

        # Create a blank image for drawing
        self.image = Image.new("L", (28, 28), 0)
        self.draw_obj = ImageDraw.Draw(self.image)
        
        # Draw 28x28 grid
        self.draw_grid()

    def draw_grid(self):
        """Draws a 28x28 grid on the canvas."""
        for i in range(28):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.canvas_size, fill="gray")
            self.canvas.create_line(0, x, self.canvas_size, x, fill="gray")

    def draw(self, event):
        """Draws a black pixel where the user moves the mouse."""
        x, y = event.x // self.cell_size, event.y // self.cell_size  # Convert to 28x28 scale
        if 0 <= x < 28 and 0 <= y < 28:
            self.draw_obj.rectangle([x, y, x + 1, y + 1], fill=255)
            self.canvas.create_rectangle(
                x * self.cell_size, y * self.cell_size, 
                (x + 1) * self.cell_size, (y + 1) * self.cell_size, 
                fill="black", outline="black"
            )

    def clear_canvas(self):
        """Clears the drawing canvas and resets the image."""
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), 0)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.draw_grid()
        self.prediction_label.set("")

    def predict_digit(self):
        """Preprocesses the drawing and predicts the digit using KNN."""
        img_resized = np.array(self.image) / 255.0
        prediction = self.model.predict(img_resized)
        self.prediction_label.set(f"Predicted Digit: {prediction}")

# Run the GUI
root = tk.Tk()
root.title("Draw a Digit and Predict with KNN")
app = DigitDrawer(root, knn)
root.mainloop()