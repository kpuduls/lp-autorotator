import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt

class AutoImageRotator:
    def __init__(self, image):
        self.image = image

    def rectangle_angle_detection(self):
        angle = 0
        return angle

    def rotate_image(self):
        rotated_image = self.image
        return rotated_image

    def skew_image_to_normal(self):
        skewed_image = self.image

if __name__ == '__main__':
    # Load the image
    image = cv2.imread('image3.jpg')
    original_image_with_rectangles = image.copy()  # Create a copy of the original image to draw rectangles on

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(blurred, contours, -1, (0, 255, 0), 3)
    # Display the original image with filtered rectangles and tilt line
    cv2.imshow('contours', blurred)

    # Create a list to store areas of rectangles and their centroids
    rectangle_centroids = []

    # Calculate the area of the whole image
    total_area = image.shape[0] * image.shape[1]

    # Iterate through contours to find rotated bounding box
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        area = rect[1][0] * rect[1][1]  # Calculate area from width and height of the rectangle

        # Exclude areas that are either close to full image or very small
        if total_area * 0.01 <= area <= total_area * 0.05:
            # Get the centroid of the rectangle
            centroid = np.mean(contour, axis=0)[0]
            rectangle_centroids.append(centroid)

            # Draw the rectangle on the original image
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(original_image_with_rectangles, [box], 0, (0, 255, 0), 2)

    # Calculate the tilt angle assuming rectangles are on a linear line
    if len(rectangle_centroids) > 1:
        centroids_array = np.array(rectangle_centroids)
        vx, vy, x, y = cv2.fitLine(centroids_array, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate the angle in degrees
        angle = float(np.arctan2(vy, vx) * 180 / np.pi)

        # Ensure the angle is in the range [0, 360)
        angle %= 360

        # Calculate start and end points for the line
        lefty = int((-x * vy / vx) + y)
        righty = int(((image.shape[1] - x) * vy / vx) + y)
        cv2.line(original_image_with_rectangles, (image.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 2)

        # Calculate perpendicular distances of rectangle centroids from the line
        distances = np.abs(
            (vx * (centroids_array[:, 1] - y) - vy * (centroids_array[:, 0] - x)) / np.sqrt(vx ** 2 + vy ** 2))

        # Calculate the mean and standard deviation of distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Define a threshold for outlier removal (e.g., 2 standard deviations)
        threshold = 1 * std_distance

        # Remove rectangles whose centroid errors are outliers
        filtered_rectangles = []
        filtered_distances = []
        for centroid, distance in zip(rectangle_centroids, distances):
            if distance <= mean_distance + threshold:
                filtered_rectangles.append(centroid)
                filtered_distances.append(distance)

        # Draw the filtered rectangles on the image
        for centroid in filtered_rectangles:
            cv2.circle(original_image_with_rectangles, tuple(map(int, centroid)), 3, (255, 0, 0), -1)

        # Fit a new line with the filtered rectangle centroids
        if len(filtered_rectangles) > 1:
            filtered_centroids_array = np.array(filtered_rectangles)
            vx_new, vy_new, x_new, y_new = cv2.fitLine(filtered_centroids_array, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty_new = int((-x_new * vy_new / vx_new) + y_new)
            righty_new = int(((image.shape[1] - x_new) * vy_new / vx_new) + y_new)
            cv2.line(original_image_with_rectangles, (image.shape[1] - 1, righty_new), (0, lefty_new), (255, 0, 0), 2)

            # Recalculate the tilt angle of the new line
            angle_new = float(np.arctan2(vy_new, vx_new) * 180 / np.pi)
            angle_new %= 360  # Ensure the angle is in the range [0, 360)

            # Create a rotation matrix to rotate the image back
            center = (image.shape[1] // 2, image.shape[0] // 2)
            if angle_new < 180:
                rotation_matrix = cv2.getRotationMatrix2D(center, angle_new, 1.0)  # Clockwise rotation
            else:
                rotation_matrix = cv2.getRotationMatrix2D(center, -360 + angle_new, 1.0)  # Counterclockwise rotation

            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

            # Display the rotated image
            cv2.imshow('Rotated Image', rotated_image)

        else:
            angle_new = 0

    else:
        angle = 0
        angle_new = 0

    # Display the original image with filtered rectangles and tilt line
    cv2.imshow('Original Image with Rectangles and Tilt Line', original_image_with_rectangles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Original Tilt angle:", angle)
    print("New Tilt angle:", angle_new)
