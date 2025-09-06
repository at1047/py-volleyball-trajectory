import cv2
import numpy as np
import os

def overlay_image(background, overlay, x, y):
    """
    Overlay an image with transparency (PNG) onto a background image
    """
    # Get the dimensions of the overlay image
    h, w = overlay.shape[:2]
    
    # Create a region of interest (ROI) in the background image
    roi = background[y:y+h, x:x+w]
    
    # If the overlay has an alpha channel
    if overlay.shape[2] == 4:
        # Split the overlay into color and alpha channels
        overlay_rgb = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0
        
        # Reshape alpha for broadcasting
        alpha = np.expand_dims(alpha, axis=-1)
        
        # Blend the images
        blended = roi * (1 - alpha) + overlay_rgb * alpha
        background[y:y+h, x:x+w] = blended
    else:
        # If no alpha channel, just copy the overlay
        background[y:y+h, x:x+w] = overlay
    
    return background

def increase_brightness(img, value=30):
    # Split the image into BGR and alpha channels
    bgr = img[:, :, :3]
    alpha = img[:, :, 3] if img.shape[2] == 4 else None
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Increase brightness
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    # Merge HSV channels and convert back to BGR
    final_hsv = cv2.merge((h, s, v))
    bright_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    # If there was an alpha channel, add it back
    if alpha is not None:
        bright_bgr = cv2.merge((bright_bgr, alpha))
    
    return bright_bgr

# Read the images
background = cv2.imread("datasets/synthetic/assets/background.jpg")
volleyball = cv2.imread("datasets/synthetic/assets/volleyball_feather.png", cv2.IMREAD_UNCHANGED)

# Resize volleyball if needed (adjust size as needed)
volleyball = cv2.resize(volleyball, (65, 65)) # scale range: 25-65
volleyball = cv2.blur(volleyball,(7,7))

volleyball = increase_brightness(volleyball, value=20)


# Position to place the volleyball (adjust coordinates as needed)
x, y = 1000, 650

# Overlay the volleyball on the background
result = overlay_image(background.copy(), volleyball, x, y)


image = cv2.circle(result, (1970,835), radius=5, color=(0, 0, 255), thickness=-1)
image = cv2.circle(result, (110,835), radius=5, color=(0, 0, 255), thickness=-1)
image = cv2.circle(result, (750,650), radius=5, color=(0, 0, 255), thickness=-1)
image = cv2.circle(result, (1350,650), radius=5, color=(0, 0, 255), thickness=-1)


output_dir = 'datasets/synthetic/Pictures'
os.makedirs(output_dir, exist_ok=True)
    
output_path = os.path.join(output_dir, f'synthetic_frame.jpg')
cv2.imwrite(output_path, result)

print("Overlay complete. Saved as 'synthetic_frame.jpg'")