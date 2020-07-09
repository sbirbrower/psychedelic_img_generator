import numpy as np
import cv2
from scipy.ndimage import generic_filter
from scipy.stats import entropy
from skimage import filters

strel = np.array([[0,   0,   1,   1,   1,   0,   0],
                  [0,   1,   1,   1,   1,   1,   0],
                  [1,   1,   1,   1,   1,   1,   1],
                  [1,   1,   1,   1,   1,   1,   1],
                  [1,   1,   1,   1,   1,   1,   1],
                  [0,   1,   1,   1,   1,   1,   0],
                  [0,   0,   1,   1,   1,   0,   0]])

log_lookup = {x: np.log(x) if x != 0 else None for x in [x for x in range(257)]} 


def fast_entropy(values):
    """
    attempt at making -sum(pk * log(pk)) --> (scipy's entropy function) 
    faster using a lookup table of length 256 to avoid recalculating
    the logarithms. unfortunately it is not faster :(
     
    CPU times: user 1min 12s, sys: 268 ms, total: 1min 13s
    Wall time: 1min 13s
    """
    probabilities, _ = np.histogram(values.ravel(), 256, [0,256])
    
    probabilities = probabilities[probabilities != 0]
    u, inv = np.unique(probabilities, return_inverse=True)
    log_array = np.array([log_lookup[x] for x in u])[inv].reshape(probabilities.shape)

    prob_sum = np.sum(probabilities)
    return -1 * np.sum((probabilities / prob_sum) * (log_array - np.log(prob_sum)))

def _entropy(values): 
    """
    it seems like the built in entropy function is still faster
    
    CPU times: user 48.2 s, sys: 64.8 ms, total: 48.3 s
    Wall time: 48.3 s
    """
    probabilities, _ = np.histogram(values.ravel(), 256, [0,256])
    return entropy(probabilities)

def entropyfilt(img, footprint): 
    """
    returns 1D array where each pixel is the calculted entropy
    value of its local neighborhood defined by footprint
    """
    return generic_filter(img.astype(np.float), _entropy, footprint=strel)


def process_image(img_path):

    img = cv2.imread(img_path)

    # resize the image for speed
    while img.size > 450*450*3:
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

    original_img = img.copy()

    b,g,r = cv2.split(img)


    # find the entropy of each channel in the image, then recombine. will take a while
    b_entropy = entropyfilt(b, strel)
    g_entropy = entropyfilt(g, strel)
    r_entropy = entropyfilt(r, strel)

    img_entropy = cv2.merge((b_entropy, g_entropy, r_entropy))
    img_entropy = cv2.normalize(img_entropy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # enhance the colors using linear stretch
    enhanced_img = img_entropy / 255

    upper = 0.85
    lower = 0.75
    enhanced_img = (enhanced_img - lower) * (255 / (upper - lower))

    np.clip(enhanced_img, 240, 255, enhanced_img)

    # then invert the image for cooler colors
    enhanced_img = 1 - enhanced_img 
    enhanced_img = cv2.normalize(enhanced_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # if there is too much white in the photo, use the original entropy image to fill it in
    if (np.sum(enhanced_img == 255) / enhanced_img.size) > 0.6:
        
        # convert image into hsv, up the saturation, convert back
        hsv = cv2.cvtColor(img_entropy, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s += 10 

        hsv = cv2.merge((h, s, v))

        saturated_entr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # invert colors and blur the image
        saturated_entr = 1 - saturated_entr 

        saturated_entr = cv2.normalize(saturated_entr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # find forground and background to partially fill in the image with masking
        threshold = filters.threshold_otsu(saturated_entr) 

        saturated_entr[saturated_entr > 1.3 * threshold] = 255
        saturated_entr[saturated_entr < 1.3 * threshold] = 0
        saturated_entr = cv2.bitwise_not(saturated_entr)

        # switch red and blue color channels for cleaner look
        red = saturated_entr[:,:,2].copy()
        blue = saturated_entr[:,:,0].copy()

        saturated_entr[:,:,0] = red
        saturated_entr[:,:,2] = blue

        image = cv2.bitwise_and(enhanced_img, saturated_entr)

        
    else:
        image = enhanced_img


    # extract the black from the colored image, we will add texture to this

    BLACK = 0
    WHITE = 255

    mask = image.copy()
    black_color_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    black_color_img[black_color_img != BLACK] = WHITE

    # dilate the black to filter out noise but maintain shapes
    kernel = np.ones((4, 4), np.uint8)
    black_color_img = cv2.dilate(black_color_img, kernel, iterations=1).astype(np.uint8)

    black_color_img = cv2.bitwise_not(black_color_img)

    # find contours in black_color_img, only fill the biggest 6 contours in 

    contours, _ = cv2.findContours(black_color_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_areas = sorted(contours, key=cv2.contourArea)
    mask1 = np.zeros(black_color_img.shape, np.uint8)


    shape1 = cv2.drawContours(mask1, [largest_areas[-1]], 0, (255,255,255,255), -1)
    shape2 = cv2.drawContours(mask1, [largest_areas[-2]], 0, (255,255,255,255), -1)
    shape3 = cv2.drawContours(mask1, [largest_areas[-3]], 0, (255,255,255,255), -1)
    shape4 = cv2.drawContours(mask1, [largest_areas[-4]], 0, (255,255,255,255), -1)
    shape5 = cv2.drawContours(mask1, [largest_areas[-5]], 0, (255,255,255,255), -1)
    shape6 = cv2.drawContours(mask1, [largest_areas[-6]], 0, (255,255,255,255), -1)
    shape7 = cv2.drawContours(mask1, [largest_areas[-7]], 0, (255,255,255,255), -1)

    # combine contours
    shapes = cv2.bitwise_and(shape1, shape2, shape3, shape4)
    shapes = cv2.bitwise_and(shapes, shape5, shape6, shape7)

    # color in the black shapes using color from thresholded original image
    threshold = filters.threshold_otsu(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)) 

    mask2 = img.copy()

    mask2[mask2 > 1.3 * threshold] = WHITE
    mask2[mask2 < 1.3 * threshold] = BLACK

    colored_shapes = cv2.bitwise_and(cv2.cvtColor(shapes, cv2.COLOR_GRAY2BGR), mask2)

    red = colored_shapes[:,:,2].copy()
    blue = colored_shapes[:,:,0].copy()

    colored_shapes[:,:,0] = red
    colored_shapes[:,:,2] = blue

    mask = (image == BLACK)
    final = np.copy(image)
    final[mask] = colored_shapes[mask]


    # fill in possible white sides to black by flood filling the corners

    edge_mask = final.copy()
    edge_mask = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

    GREY = 100

    hori_size, vert_size = edge_mask.shape

    if edge_mask[0][0] == WHITE:
            cv2.floodFill(edge_mask, None, (0, 0), GREY)

    if edge_mask[hori_size - 1][0] == WHITE:
        cv2.floodFill(edge_mask, None, (0, hori_size - 1), GREY)
            
    if edge_mask[hori_size - 1][vert_size - 1] == WHITE:
            cv2.floodFill(edge_mask, None, (vert_size - 1, hori_size - 1), GREY)
            
    if edge_mask[0][vert_size - 1] == WHITE:
            cv2.floodFill(edge_mask, None, (vert_size - 1, 0), GREY)

            
    edge_mask = edge_mask != GREY

    edge_mask = edge_mask.astype(np.uint8)  
    edge_mask *= 255

    kernel = np.ones((2, 2), np.uint8)
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=1).astype(np.uint8)

    final = cv2.bitwise_and(cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR), final)


    ksize = (2, 2) 
    final = cv2.blur(final, ksize)

    cv2.imwrite('/Users/sydney/Files/projects/static/final.png', final)