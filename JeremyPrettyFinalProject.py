# Jeremy Pretty Final Project
# April 2, 2023
import cv2
import os
import pytesseract

russia_1 = os.path.join(os.path.dirname(__file__), 'russia1.jpeg')
russia_2 = os.path.join(os.path.dirname(__file__), 'russia2.jpeg')
texas = os.path.join(os.path.dirname(__file__), 'texas.jpeg')
license_plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

def detect_license_plates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    license_plate_locations = license_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return license_plate_locations

def draw_boundary_boxes(image, license_plate_locations):
    for (x, y, w, h) in license_plate_locations:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image

def extract_license_plate(image, license_plate_location):
    x, y, w, h = license_plate_location
    crop = image[y:y+h, x:x+w]
    return crop

def process_license_plate(image):
    binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    return blurred

def recognize_license_plate(image):
    characters = pytesseract.image_to_string(image, lang='rus', config='--psm 6')
    return characters

def main():
    images = [(russia_1, "russia1"), (russia_2, "russia2"), (texas, "texas")]

    for image_path, image_name in images:
        image = cv2.imread(image_path)

        # Detect the license plates
        license_plate_locations = detect_license_plates(image)

        # Draw red boundary boxes around the detected plates
        image_with_boxes = draw_boundary_boxes(image.copy(), license_plate_locations)

        # Extract the license plates
        license_plates = list(map(extract_license_plate, [image]*len(license_plate_locations), license_plate_locations))

        # Process the license plates
        processed_license_plates = list(map(process_license_plate, license_plates))

        # Recognize the license plates
        license_plate_characters = list(map(recognize_license_plate, processed_license_plates))

        # Display the images with the red boundary boxes
        cv2.imshow(f"{image_name} with boundary boxes", image_with_boxes)

        # Print the detected license plate locations and characters
        print(f"License plate locations ({image_name}):")
        print(license_plate_locations)

        print(f"License plate characters ({image_name}):")
        print(license_plate_characters)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
