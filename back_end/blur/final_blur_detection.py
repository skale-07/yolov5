# From github_code_srikanth


# USAGE
# python blur_detector_image.py --image photos/resume_01.png --thresh 27

# import the necessary packages
import numpy as np
import imutils
import cv2

import matplotlib.pyplot as plt
import numpy as np
import os



def detect_blur_fft(image, size=60, thresh=10, vis=False):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))

        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


def blur_detection(image_path, thresh):
    # load the input image from disk, resize it, and convert it to
    # grayscale
    orig = cv2.imread(image_path)
    orig = imutils.resize(orig, width=500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # apply our blur detector using the FFT
    (mean, blurry) = detect_blur_fft(gray, size=60,
                                     thresh=thresh)

    return mean


def main(directory_path, thresh=20, ascending=False):
    blur_scores = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            blur_score = blur_detection(image_path, thresh)
            blur_scores.append((blur_score, image_path))

    # Sort the list of tuples based on the blur score
    blur_scores.sort(key=lambda x: x[0], reverse=not ascending)

    # Create a list of sorted image paths
    sorted_image_paths = [image_path for _, image_path in blur_scores]

    return sorted_image_paths

if __name__ == '__main__':
    directory_path = r"C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\blur\sample_photos"

    print(main(directory_path, thresh=20))


# ['C:\\Users\\tdewa\\Code\\snapsort_project\\SnapSort\\back_end\\blur\\sample_photos\\high-resolution-gfinds1akzwf6vcq.jpg',
#  'C:\\Users\\tdewa\\Code\\snapsort_project\\SnapSort\\back_end\\blur\\sample_photos\\DSC_5919-2.jpg',
#  'C:\\Users\\tdewa\\Code\\snapsort_project\\SnapSort\\back_end\\blur\\sample_photos\\DSC_6067-2.jpg',
#  'C:\\Users\\tdewa\\Code\\snapsort_project\\SnapSort\\back_end\\blur\\sample_photos\\test3.jpg',
#  'C:\\Users\\tdewa\\Code\\snapsort_project\\SnapSort\\back_end\\blur\\sample_photos\\test1.jpg',
#  'C:\\Users\\tdewa\\Code\\snapsort_project\\SnapSort\\back_end\\blur\\sample_photos\\test2.jpg',
#  'C:\\Users\\tdewa\\Code\\snapsort_project\\SnapSort\\back_end\\blur\\sample_photos\\test5.jpg',
#  'C:\\Users\\tdewa\\Code\\snapsort_project\\SnapSort\\back_end\\blur\\sample_photos\\blurry_background_test_img.jpg',
#  'C:\\Users\\tdewa\\Code\\snapsort_project\\SnapSort\\back_end\\blur\\sample_photos\\test4.jpg']






# import numpy as np
# import imutils
# import cv2
# import matplotlib.pyplot as plt
# import os
#
# def detect_blur_fft(image, size=60, thresh=10, vis=False):
#     (h, w) = image.shape
#     (cX, cY) = (int(w / 2.0), int(h / 2.0))
#
#     fft = np.fft.fft2(image)
#     fftShift = np.fft.fftshift(fft)
#
#     if vis:
#         magnitude = 20 * np.log(np.abs(fftShift))
#         (fig, ax) = plt.subplots(1, 2)
#         ax[0].imshow(image, cmap="gray")
#         ax[0].set_title("Input")
#         ax[0].set_xticks([])
#         ax[0].set_yticks([])
#
#         ax[1].imshow(magnitude, cmap="gray")
#         ax[1].set_title("Magnitude Spectrum")
#         ax[1].set_xticks([])
#         ax[1].set_yticks([])
#
#         plt.show()
#
#     fftShift[cY - size:cY + size, cX - size:cX + size] = 0
#     fftShift = np.fft.ifftshift(fftShift)
#     recon = np.fft.ifft2(fftShift)
#
#     magnitude = 20 * np.log(np.abs(recon))
#     mean = np.mean(magnitude)
#
#     return (mean, mean <= thresh)
#
# def blur_detection(image_path, thresh):
#     orig = cv2.imread(image_path)
#     orig = imutils.resize(orig, width=500)
#     gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
#
#     # Edge detection to find the foreground
#     edged = cv2.Canny(gray, 50, 100)
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     mask = np.zeros(gray.shape, dtype="uint8")
#
#     if len(contours) > 0:
#         c = max(contours, key=cv2.contourArea)
#         cv2.drawContours(mask, [c], -1, 255, -1)
#
#     foreground = cv2.bitwise_and(gray, gray, mask=mask)
#     background = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask))
#
#     (mean_foreground, blurry_foreground) = detect_blur_fft(foreground, size=60, thresh=thresh)
#     (mean_background, blurry_background) = detect_blur_fft(background, size=60, thresh=thresh)
#
#     # Define criteria for considering an image blurry
#     if not blurry_foreground and blurry_background:
#         return (mean_foreground, mean_background, image_path, False)
#     else:
#         return (mean_foreground, mean_background, image_path, True)
#
# def main(directory_path, thresh=20, ascending=False):
#     blur_scores = []
#
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(directory_path, filename)
#             mean_foreground, mean_background, image_path, blurry = blur_detection(image_path, thresh)
#             blur_scores.append((mean_foreground, mean_background, image_path, blurry))
#
#     blur_scores.sort(key=lambda x: (x[3], x[0], x[1]), reverse=not ascending)
#
#     sorted_image_paths = [image_path for _, _, image_path, _ in blur_scores]
#
#     return sorted_image_paths
#
# if __name__ == '__main__':
#     directory_path = r"C:\Users\tdewa\Code\snapsort_project\SnapSort\back_end\blur\sample_photos"
#     print(main(directory_path, thresh=20))
