import cv2
import numpy as np
from pyzbar.pyzbar import decode
from typing import List, Optional, Union, Any


class QRReader:
    """QR code reader class for detecting and drawing QR codes in images."""
    
    def __init__(self):
        """Initialize the QR code reader with empty data history."""
        self.data_history: List[str] = ['link']  # Track recognized QR code data to avoid duplicate display

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        """Detect QR codes in the image and draw bounding boxes with content.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Image with detected QR codes highlighted
        """
        # Detect QR codes
        qr_codes = decode(img)

        # If direct detection fails, try enhancing the image and detect again
        if not qr_codes:
            enhanced_img = self._enhance_image(img)
            qr_codes = decode(enhanced_img)

        # Process detected QR codes
        for qr in qr_codes:
            qr_data = qr.data.decode('utf-8')
            print(f"QR code detected: {qr_data}")

            # Update history only if data is different
            if qr_data != self.data_history[-1]:
                self.data_history.append(qr_data)
                print(f"History: {self.data_history[1:]}")  # Skip initial 'link'

            # Draw rectangle
            (x, y, w, h) = qr.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 5)

            # Draw QR code content text
            cv2.putText(img, qr_data, (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 1)

            # Draw QR code polygon lines
            if hasattr(qr, 'polygon') and qr.polygon:
                for i in range(len(qr.polygon)):
                    pt1 = (qr.polygon[i].x, qr.polygon[i].y)
                    pt2 = (qr.polygon[(i + 1) % len(qr.polygon)].x, qr.polygon[(i + 1) % len(qr.polygon)].y)
                    cv2.line(img, pt1, pt2, (0, 255, 0), 3)

        return img

    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        """Enhance image to improve QR code detection rate.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            Enhanced binary image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return binary


def main() -> None:
    """Main function to run the QR code scanner."""
    reader = QRReader()

    # Choose between camera or image file
    use_camera = False  # Set to False to read from image file

    if use_camera:
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture camera image")
                break

            # Detect and draw QR codes
            output_img = reader.detect_and_draw(img)

            cv2.imshow('qr_detector Scanner', output_img)

            # Exit on ESC key press
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
    else:
        # Read image file
        img_path = 'qr_code.jpg'  # Replace with your image path
        img = cv2.imread(img_path)
        if img is not None:
            output_img = reader.detect_and_draw(img)
            cv2.imshow('qr_detector Scanner', output_img)
            print("Press any key to close window")
            cv2.waitKey(0)
        else:
            print(f"Failed to read image file: {img_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
