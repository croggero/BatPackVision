import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time

counter = 0


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # recent polynomial coefficients
        self.recent_fit = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # counter to reset after 5 iterations if issues arise
        self.counter = 0


def pipeline(img, s_thresh=(0, 60), r_thresh=(100, 255)):
    dst = np.copy(img)

    r_channel = dst[:, :, 0]
    g_channel = dst[:, :, 1]
    b_channel = dst[:, :, 2]

    hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS)

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 255

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255

    ret, otsu = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    combined = np.zeros_like(s_binary)
    combined[(s_binary == 255) | (otsu == 255)] = 255
    combined = cv2.bitwise_not(otsu)

    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.erode(combined, kernel, iterations=5)
    # cv2.imshow("Test", combined)

    contour_image, contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    binary_contour_image = np.zeros_like(contour_image)
    cv2.drawContours(binary_contour_image, [c], -1, (255, 255, 255), 2)

    x_size = binary_contour_image.shape[0] - 1
    y_size = binary_contour_image.shape[1] - 1

    binary_contour_image[[0, 1, 2, 3, 4,
                          x_size, x_size - 1, x_size - 2, x_size - 3, x_size - 4], :] = 0


    return binary_contour_image


def birds_eye(img):
    binary_img = pipeline(img)

    img_size = (binary_img.shape[1], binary_img.shape[0])
    src = np.float32([[1280, 100], [1280, img_size[1]], [0, img_size[1]], [0, 100]])

    offset = 4  # offset for dst points
    dst = np.float32([[img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]],
                      [offset, img_size[1]], [offset, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    top_down = cv2.warpPerspective(binary_img, M, img_size)

    return top_down, M


def count_check(line):
    if line.counter >= 5:
        line.detected = False


def first_lines(img):
    # Load the birds eye image and transform matrix from birds_eye
    binary_warped, perspective_M = birds_eye(img)

    # Histogram of the bottom half of the image
    histogram = np.sum(binary_warped[(int)(binary_warped.shape[0] / 5):, :], axis=0)

    # Output image an to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the right and left halves of the histogram
    # These will be the starting point for the right and left lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 15

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    rightx_current = rightx_base
    leftx_current = leftx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive right and left lane pixel indices
    right_lane_inds = []
    left_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and left and right)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        # Append these indices to the lists
        right_lane_inds.append(good_right_inds)
        left_lane_inds.append(good_left_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

    # Concatenate the arrays of indices
    right_lane_inds = np.concatenate(right_lane_inds)
    left_lane_inds = np.concatenate(left_lane_inds)

    # Extract right and left line pixel positions
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    # Fit a second order polynomial to each
    # The challenge videos sometimes throw errors, so the below try first
    # Upon the error being thrown, set line.detected to False
    # right line first
    try:
        n = 5
        right_line.current_fit = np.polyfit(righty, rightx, 2)
        right_line.all_x = rightx
        right_line.all_y = righty
        right_line.recent_fit.append(right_line.current_fit)
        if len(right_line.recent_fit) > 1:
            right_line.diffs = (right_line.recent_fit[-2] - right_line.recent_fit[-1]) / right_line.recent_fit[-2]
        right_line.recent_fit = right_line.recent_fit[-n:]
        right_line.best_fit = np.mean(right_line.recent_fit, axis=0)
        right_fit = right_line.current_fit
        right_line.detected = True
        right_line.counter = 0
    except TypeError:
        right_fit = right_line.best_fit
        right_line.detected = False
    except np.linalg.LinAlgError:
        right_fit = right_line.best_fit
        right_line.detected = False

    # Next, left line
    try:
        n = 5
        left_line.current_fit = np.polyfit(lefty, leftx, 2)
        left_line.all_x = leftx
        left_line.all_y = lefty
        left_line.recent_fit.append(left_line.current_fit)
        if len(left_line.recent_fit) > 1:
            left_line.diffs = (left_line.recent_fit[-2] - left_line.recent_fit[-1]) / left_line.recent_fit[-2]
        left_line.recent_fit = left_line.recent_fit[-n:]
        left_line.best_fit = np.mean(left_line.recent_fit, axis=0)
        left_fit = left_line.current_fit
        left_line.detected = True
        left_line.counter = 0
    except TypeError:
        left_fit = left_line.best_fit
        left_line.detected = False
    except np.linalg.LinAlgError:
        left_fit = left_line.best_fit
        left_line.detected = False


def second_ord_poly(line, val):
    a = line[0]
    b = line[1]
    c = line[2]
    formula = (a * val ** 2) + (b * val) + c

    return formula


def draw_lines(img):
    # Pull in the image
    binary_warped, perspective_M = birds_eye(img)

    # Check if lines were last detected; if not, re-run first_lines
    if right_line.detected == False | left_line.detected == False:
        first_lines(img)

    global counter
    counter += 1
    if counter == 3:
        first_lines(img)
        counter = 0

    # Set the fit as the current fit for now
    right_fit = right_line.current_fit
    left_fit = left_line.current_fit

    # Again, find the lane indicators
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    if len(right_fit) >= 2:
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

    # Set the x and y values of points on each line
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    # Fit a second order polynomial to each again.
    # Similar to first_lines, need to try in case of errors
    # right line first
    try:
        n = 5
        right_line.current_fit = np.polyfit(righty, rightx, 2)
        right_line.all_x = rightx
        right_line.all_y = righty
        right_line.recent_fit.append(right_line.current_fit)
        if len(right_line.recent_fit) > 1:
            right_line.diffs = (right_line.recent_fit[-2] - right_line.recent_fit[-1]) / right_line.recent_fit[-2]
        right_line.recent_fit = right_line.recent_fit[-n:]
        right_line.best_fit = np.mean(right_line.recent_fit, axis=0)
        right_fit = right_line.current_fit
        right_line.detected = True
        right_line.counter = 0
    except TypeError:
        right_fit = right_line.best_fit
        count_check(right_line)
    except np.linalg.LinAlgError:
        right_fit = right_line.best_fit
        count_check(right_line)

    # Now left line
    try:
        n = 5
        left_line.current_fit = np.polyfit(lefty, leftx, 2)
        left_line.all_x = leftx
        left_line.all_y = lefty
        left_line.recent_fit.append(left_line.current_fit)
        if len(left_line.recent_fit) > 1:
            left_line.diffs = (left_line.recent_fit[-2] - left_line.recent_fit[-1]) / left_line.recent_fit[-2]
        left_line.recent_fit = left_line.recent_fit[-n:]
        left_line.best_fit = np.mean(left_line.recent_fit, axis=0)
        left_fit = left_line.current_fit
        left_line.detected = True
        left_line.counter = 0
    except TypeError:
        left_fit = left_line.best_fit
        count_check(left_line)
    except np.linalg.LinAlgError:
        left_fit = left_line.best_fit
        count_check(left_line)

    # Generate x and y values for plotting
    fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]
    fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # Color in right and left line pixels
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]

    # Calculate the pixel curve radius
    y_eval = np.max(fity)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 15 / 720  # meters per pixel in y dimension
    xm_per_pix = 2 / 1200  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    right_fit_cr = np.polyfit(right_line.all_y * ym_per_pix, right_line.all_x * xm_per_pix, 2)
    left_fit_cr = np.polyfit(left_line.all_y * ym_per_pix, left_line.all_x * xm_per_pix, 2)

    # Calculate the new radii of curvature
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    avg_rad = round(np.mean([right_curverad, left_curverad]), 0)
    rad_text = "Radius of Curvature = {}(m)".format(avg_rad)

    # Calculating middle of the image, aka where the user camera is
    middle_of_image = img.shape[1] / 2
    user_position = middle_of_image * xm_per_pix

    # Calculating middle of the lane
    right_line_base = second_ord_poly(right_fit_cr, img.shape[0] * ym_per_pix)
    left_line_base = second_ord_poly(left_fit_cr, img.shape[0] * ym_per_pix)
    lane_mid = (right_line_base + left_line_base) / 2

    # Calculate distance from center and list differently based on right or left
    dist_from_center = lane_mid - user_position
    if dist_from_center > 0:
        txt = "Left Grass: {} ft".format(dist_from_center * 3.28)
    else:
        txt = "Right Grass: {} ft".format(dist_from_center * 3.28)

    # List user's position in relation to middle on the image and radius of curvature
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, txt, (10, 200), font, 1, (255, 255, 255), 2)

    # Invert the transform matrix from birds_eye (to later make the image back to normal below)
    Minv = np.linalg.inv(perspective_M)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_right = np.array([np.transpose(np.vstack([fit_rightx, fity]))])
    pts_left = np.array([np.flipud(np.transpose(np.vstack([fit_leftx, fity])))])
    pts = np.hstack((pts_right, pts_left))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result


def process_image(image):
    result = draw_lines(image)

    return result


right_line = Line()
left_line = Line()

cap = cv2.VideoCapture('Sequence3.mp4')
fps = 0
ret, frame = cap.read()
while ret:
    start = time.time()
    ret, frame = cap.read()
    image = process_image(frame)
    end = time.time()

    fps = 1 / (end - start)
    fps_text = "{} fps".format(round(fps, 2))
    # cv2.putText(image, fps_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('frame', image)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
