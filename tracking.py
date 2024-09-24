import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load video
# Path 1
#cap = cv2.VideoCapture('video_paths/path1.mp4')

# Path 2
cap = cv2.VideoCapture('video_paths/path2.mp4')

# List to store microrobot coordinates
coordinates = []
frame_count = 0
save_interval = 100  # Save an image every 100 frames

# Read frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply circular mask
    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = 160  # Adjust the ROI circular radius (circular petri dish)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply the mask to make everything outside the circle black
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Set everything outside the circle to black by inverting the mask
    outside_black = cv2.bitwise_not(mask)
    frame_with_black_background = cv2.add(masked, outside_black)
    
    # Thresholding
    _, thresh = cv2.threshold(frame_with_black_background, 100, 255, cv2.THRESH_BINARY)
    
    # Invert the threshold to make the small black point become white
    inverted_thresh = cv2.bitwise_not(thresh)
    
    # Find contours in the inverted image
    contours, _ = cv2.findContours(inverted_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assuming the microrobot is the smallest contour
        c = min(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coordinates.append((cX, cY))
            
            # Draw the contour and its center on the original frame
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

            # Get the bounding box (object width and height)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            print(f"Frame {frame_count}: Width={w}, Height={h}")

            # Save frames with the detected microrobot
            if frame_count % save_interval == 0:
                frame_filename = f'processed_frame_path2_{frame_count}.png'
                cv2.imwrite(frame_filename, frame)
                print(f'Frame {frame_count} saved as {frame_filename}')
    
    # Save the thresholded frame to visualize the processing
    if frame_count % save_interval == 0:
        thresh_filename = f'thresh_frame_path2_{frame_count}.png'
        cv2.imwrite(thresh_filename, inverted_thresh)
        print(f'Threshold frame {frame_count} saved as {thresh_filename}')
        
cap.release()

# Save the coordinates to plot later
if coordinates:
    x_coords, y_coords = zip(*coordinates)
    
    # Set a larger figure size
    plt.figure(figsize=(10, 6))
    
    # Plot the path with custom markers and line style
    plt.plot(x_coords, y_coords, linestyle='-', marker='o', color='#1f77b4', markersize=5, markerfacecolor='red', markeredgecolor='black', linewidth=2)
    
    # Add grid to make it easier to read
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
    
    # Set title and axis labels with larger font size
    plt.title('Microrobot Path 2', fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate (pixels)', fontsize=14)
    plt.ylabel('Y Coordinate (pixels)', fontsize=14)
    
    # Set limits to avoid points sticking to the edges
    plt.xlim(min(x_coords) - 10, max(x_coords) + 10)
    plt.ylim(min(y_coords) - 10, max(y_coords) + 10)
    
    # Customize tick size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the plot to a file
    plt.savefig('trajectory_plot_path2.png', bbox_inches='tight', dpi=300)
    print('Path saved as trajectory_plot.png')
