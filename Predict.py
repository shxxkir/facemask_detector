from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('FaceMask_model.h5')

# Load test image
img_path = 'Test Images/test1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
prediction = float(prediction)   # Convert to a simple float!

# Determine label and probability
if prediction > 0.5:
    label = f"No Mask Detected ({prediction*100:.2f}% confidence)"
else:
    label = f"Mask Detected ({(1 - prediction)*100:.2f}% confidence)"

# Plot the image with the prediction label
plt.figure(figsize=(6,6))
plt.imshow(image.load_img(img_path))  # Show original image
plt.title(label, fontsize=16)
plt.axis('off')
plt.show()