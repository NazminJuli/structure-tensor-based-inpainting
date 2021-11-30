import numpy as np
import skimage
from skimage.io import imread, imsave, imshow
from skimage.color import rgba2rgb, convert_colorspace
from skimage.util import img_as_ubyte
from skimage import segmentation
from skimage.transform import resize
from skimage.filters import threshold_local
from inpainter import Inpainter
import cv2


def extract_skin_patch(original_image):
    img_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    # define range of red color in HSV
    lower_red = np.array([0, 200, 200])
    upper_red = np.array([179, 255, 255])
    # mask1 = cv2.inRange(img_hsv, (170, 50, 20), (180, 255, 255))
    # mask2 = cv2.inRange(img_hsv, (180, 50, 20), (255, 255, 255))

    ## Merge the mask and crop the red regions
    # mask = cv2.bitwise_or(mask1, mask2)
    # croped = cv2.bitwise_and(original_image, original_image, mask=mask)
    red_mask = cv2.inRange(img_hsv, lower_red, upper_red)
    red = cv2.bitwise_and(original_image, original_image, mask=red_mask)
    # captured_frame_lab_red = cv2.inRange(original_image, np.array([0, 0, 240]), np.array([0, 0, 255]))

    cv2.imwrite('color.png', red_mask)
    cv2.imwrite('color1.png', red)
    cnts, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype='uint8')
    # red_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype='uint8')
    print("mask contour len initial:", len(cnts))

    for c in cnts:
        img_mask = cv2.fillPoly(img_mask, pts=[c], color=(255, 255, 255))
    cv2.imwrite('marked_mask.png', img_mask)
    kernel = np.ones((3, 3), 'uint8')
    skin_map = cv2.dilate(img_mask, kernel, iterations=2)
    return skin_map


def main():

    # original_image = imread("/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/OK/12.png")
    # original_mask = imread('/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/marked_mask_enlarged.png', as_gray=True)
    # original_image = imread("/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/beautiful-caucasian-woman-with-makeup.png")
    # original_mask = imread('/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/marked_mask_before.png', as_gray=True)
    # print(" image type:", original_image.dtype)
    # original_image = imread("/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/image8.jpg")
    # original_mask = imread('/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/mask8.jpg', as_gray=True)
    # img = np.array(original_image).astype('uint8')
    # img = img_as_ubyte(original_image)
    original_image = cv2.imread("/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/raw_marked.png")
    # img = original_image[:, :, ::-1]         #convert BGR to RGB
    # original_image = img

    original_mask = extract_skin_patch(original_image)
    # print(original_mask.shape, 'mask shape....')

    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    if original_image is not None and original_mask is not None:
        # print('got inputs', original_image.shape)
        h, w, ch = original_image.shape
        print(original_image.shape, original_mask.shape)
        if ch == 4:
            original_image = rgba2rgb(original_image)
            original_image = img_as_ubyte(original_image)

            # print(original_image.shape, "rgb")
        # th, tw = 400, 800
        image = original_image
        mask = original_mask
        # mask = img_as_ubyte(original_mask)
        # original_mask = threshold_local(original_mask, block_size= 35, offset=10)
        # print(" image type again:", original_mask.shape)
        # mask = image > original_mask
        # mask[original_mask != 0] = 255
        # print(" image type again:", original_image.dtype)
        # image = resize(image, (th, tw), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
        # mask = resize(mask, (th, tw), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')

        print("original image showing......")

        output_image = Inpainter(
            image,
            mask,
            patch_size = 7,
            plot_progress = True
        ).inpaint()
        # result = resize(output_image, (th, tw), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
        # img = original_image[:, :, ::-1]  # convert BGR to RGB
        # output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.png', output_image)

    else:
        print('error in loading image....')


if __name__ == '__main__':
    main()