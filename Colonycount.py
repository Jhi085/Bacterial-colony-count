import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("C:/Users/Jonathan/OneDrive - UiT Office 365/Documents/Paper 1 final data_X/Picture/Low_pH_repeat_afterreview/Repeat low pH_24h/Input/24h_lowpH_Plate-10.PNG")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to create a binary image
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize counts for different size categories
small_count = 0
medium_count = 0
large_count = 0

# Loop over the contours
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Filter out contours with small areas (noise)
    if area < 40:
        continue

    # Categorize contours based on their area
    if area < 200:
        size_category = "Small"
        color = (0, 255, 0)  # Green for small objects
        small_count += 1
    elif area < 400:
        size_category = "Medium"
        color = (0, 0, 255)  # Red for medium objects
        medium_count += 1
    else:
        size_category = "Large"
        color = (255, 0, 0)  # Blue for large objects
        large_count += 1

    # Draw the contour and size value on the original image
    cv2.drawContours(image, [contour], -1, color, 2)
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(image, f'{size_category} ({area:.0f})', (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Calculate total number of objects detected
total_objects = small_count + medium_count + large_count

# Print the number of objects detected in each size category
print("Number of small objects:", small_count)
print("Number of medium objects:", medium_count)
print("Number of large objects:", large_count)
print("Total number of objects detected:", total_objects)

# Create a folder to save the counted image if it doesn't exist
folder_path = "C:/Users/Jonathan/OneDrive - UiT Office 365/Documents/Paper 1 final data_X/Picture/Low_pH_repeat_afterreview/Repeat low pH_24h/Input/Output/counted_image"
os.makedirs(folder_path, exist_ok=True)

# Save the counted image in the folder
cv2.imwrite(os.path.join(folder_path, "counted_image.png"), image)

# Display the result
cv2.imshow('Objects Counted', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot the counts as a bar chart
categories = ['Small', 'Medium', 'Large']
counts = [small_count, medium_count, large_count]

plt.bar(categories, counts, color=['green', 'red', 'blue'])
plt.xlabel('Object Size Category')
plt.ylabel('Count')
plt.title('Object Count by Size Category')
plt.show()
