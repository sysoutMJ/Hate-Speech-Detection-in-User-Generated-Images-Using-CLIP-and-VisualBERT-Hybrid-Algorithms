# importing OpenCV(cv2) module
import cv2 

image_path = r"K:\Thesis\Facebook Hateful Meme Dataset\data\img\01247.png"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    
    blurred_image = cv2.GaussianBlur(image, (29, 29), 0)
    
    output_path = r"K:\Official_Thesis\BlurredImages\01247_blurred.png"
    cv2.imwrite(output_path, blurred_image)
    
    print(f"Blurred image saved successfully at: {output_path}")
    
    """
    # 29 is the max 
    Gaussian = cv2.GaussianBlur(image, (29, 29), 0) 
    cv2.imshow('Gaussian Blurring', Gaussian) 
    cv2.waitKey(0) # To exit window that was created to show the image. May press any keys.
    cv2.destroyAllWindows()  
    
    # Status: Working. Blurs the image to the point of no one can read it.
    """

    
    """
    cv2.imshow('Original Image', sampleHatespeechImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Status: Worked for showing the original image
    """


