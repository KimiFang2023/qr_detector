import os
import cv2
import webbrowser
from pyzbar import pyzbar
from typing import List, Any, Optional


def decode(image: np.ndarray) -> List[Any]:
    """Decode QR codes from the input image using pyzbar.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        List of decoded QR code objects
    """
    # Find and decode all barcodes in the image
    decoded_objects = pyzbar.decode(image)
    return decoded_objects


def main() -> None:
    """Main function for real-time QR code detection and decoding using webcam."""
    # Open webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Store decoded QR code data to avoid repeated processing
    decoded_data: List[str] = ["link"]  # Initial placeholder
    
    print("qr_detector Scanner started. Press 'ESC' to exit.")
    
    try:
        while True:
            # Capture frame-by-frame
            success, frame = cap.read()
            
            if not success:
                print("Warning: Failed to capture image frame.")
                break
            
            # Decode QR codes in the frame
            qr_codes = decode(frame)
            
            # Process each detected QR code
            for qr_code in qr_codes:
                # Extract and decode the data
                qr_data = qr_code.data.decode('utf-8')
                print(f"Decoded QR code data: {qr_data}")
                
                # Process new QR codes (avoid duplicates)
                if qr_data != decoded_data[-1]:
                    decoded_data.append(qr_data)
                    # Uncomment the line below to open the URL in a browser
                    # webbrowser.open(qr_data)
                    print(f"Updated data list: {decoded_data}")
                
                # Draw bounding box and text on the frame
                rect = qr_code.rect
                cv2.rectangle(
                    frame,
                    (rect[0], rect[1]),
                    (rect[0] + rect[2], rect[1] + rect[3]),
                    (255, 255, 0),
                    5
                )
                cv2.putText(
                    frame,
                    qr_data,
                    (rect[0], rect[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (255, 0, 255),
                    1
                )
            
            # Display the resulting frame
            cv2.imshow('qr_detector Scanner', frame)
            
            # Exit if 'ESC' key is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                print("Exiting qr_detector Scanner...")
                break
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()