import random
import numpy as np
import cv2
import glob
import os
import shutil

#TODO:  Restore the color lines and points at smaller size
#TODO:  Render the reproductions at the right scale
#TODO:  Write the error metrics to a file for auditing

def main():
    max_images = 100
    rows = 7
    cols = 10
    square_edge_length = 0.02223   # meters
    print_error_assessment = True                    # goodness of fit for quality control
    print_checkerboard_images = True                 # reconstructions for quality control
    camera_name = "camera2"
    folder_with_cal_images = camera_name + "\\"

    # Camera pixel resolution
    resolution = [1280,800]
    if camera_name == "camera1":
        resolution = [1280,720]
    if camera_name == "camera2":
        resolution = [1280, 720]
    if camera_name == "camera3":
        resolution = [1920, 1080]
    # Initialize arrays and constants
    object_points = []
    image_points = []
    image_size = None
    pattern_size = (rows, cols)

    # Create a matrix of XYZ triples for each corner on the checkerboard
    x = np.arange(pattern_size[0])
    y = np.arange(pattern_size[1])
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.zeros_like(xgrid)
    object_points_3D = square_edge_length * np.dstack((xgrid, ygrid, zgrid)).reshape((-1, 1, 3)).astype(np.float32)

    # Detect calibration images
    images = glob.glob(folder_with_cal_images + '*.jpg')

    if print_checkerboard_images:
        # make a directory tree for saving reconstruction images
        if os.path.isdir(folder_with_cal_images + 'checkerboard'):
            shutil.rmtree(folder_with_cal_images + 'checkerboard')
            os.makedirs(folder_with_cal_images + 'checkerboard')
        else:
            os.makedirs(folder_with_cal_images + 'checkerboard')

    if print_error_assessment:
        # make a directory tree for storing goodness of fit metrics
        if os.path.isdir(folder_with_cal_images + 'error'):
            shutil.rmtree(folder_with_cal_images + 'error')
            os.makedirs(folder_with_cal_images + 'error')
        else:
            os.makedirs(folder_with_cal_images + 'error')

    # Criteria for subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # Image number index
    index = -1
    cal_image_name = []
    good_image = []

    for filename in images:
        # Indices for post-calibration error calculations
        index = index + 1
        cal_image_name.append(filename)

        # Get an image from the file space
        rgb = cv2.imread(filename)
        if rgb is None:
            print('warning: error getting data from file, ending')
            break

        # Convert to gray scale
        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb

        # Get the image size
        cursize = gray.shape[::-1]
        print(cursize)
        if image_size is None:
            image_size = cursize
            #print('Received frame of size {}x{}'.format(*image_size))
        else:
            assert image_size == cursize

        # Use an openCV method to find the corners on the chessboard
        #     retval is a boolean.  0 for bad image.  1 for good image.  Saved in a list good_image
        #     corners is a matrix of detected interior black-on-black corner points
        retval, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        good_image.append(retval)

        # If the corner detection was successful  (retval true)
        # then get subpixel refinement and draw the corners on the image
        if retval:
            # Only images with sub pixel refinement are used in the calibration
            # refined = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
            refined = cv2.cornerSubPix(gray, corners, (7,7), (2,2), criteria)
            # Append the matrix of object points
            object_points.append(object_points_3D)
            # Append the matrix of refined image points
            image_points.append(refined)
            # Detect and draw the corners
            cv2.drawChessboardCorners(gray, pattern_size, refined, retval)
            # Put a frame number on the image for quality control
            cv2.putText(gray, str(len(image_points)), (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.namedWindow('chessboard', cv2.WINDOW_KEEPRATIO)
            cv2.imshow("chessboard", gray)
            cv2.resizeWindow('chessboard', 640, 400)
            cv2.waitKey(500)
            # Write the image for QC and documentation
            cv2.imwrite(folder_with_cal_images + '/checkerboard/image' + str(index) + '.jpg',gray)
        else:
            pass


    print(f'Detected {len(image_points)} calibration images')
    # Enforce no more than max_images in the calibration
    if max_images != 0 and len(image_points) > max_images:
        random.shuffle(image_points)
        image_points = image_points[:max_images]
        print(f'using {len(image_points)} points')

    # Perform the calibration
    retval, K, dcoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    # Get an error metric by projection world points back onto the image
    # Compute the average norm between image points and projections
    if print_error_assessment:
        mean_error = 0
        print("Object Points:      " + str(len(object_points)))
        print("Rotation Matrix:    " + str(len(rvecs)))
        print("Translation Matrix: " + str(len(tvecs)))

        jindex = -1
        for ii in range(len(cal_image_name)):
            if good_image[ii]:
                # Get the image which matches the rvec and tvec pose data
                filename = cal_image_name[ii]
                # Load and draw the image
                rgb = cv2.imread(filename)
                # Convert to gray scale
                if len(rgb.shape) == 3:
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                else:
                    gray = rgb
                # Get the index into the rvecs and tvecs pose estimates
                jindex = jindex + 1
                # find the corners using the original algorithm and plot them
                image_points_2, _ = cv2.projectPoints(object_points[jindex], rvecs[jindex], tvecs[jindex], K, dcoeffs)
                # Draw the projected points onto the image
                cv2.drawChessboardCorners(gray, pattern_size, image_points_2, True)
                cv2.putText(rgb, str(len(image_points)), (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.namedWindow('chessboard', cv2.WINDOW_KEEPRATIO)
                cv2.imshow("chessboard", gray)
                cv2.resizeWindow('chessboard', 640, 400)
                # Calculate the error and send to screen
                error = cv2.norm(image_points[jindex], image_points_2, cv2.NORM_L2) / len(image_points_2)
                print("Image:  " + str(ii) + '  Error: ' + str(error))
                mean_error += error
                # Write the image for QC and documentation
                cv2.imwrite(folder_with_cal_images + '/error/image' + str(ii) + '.jpg',gray)
                cv2.waitKey(500)
        print("total error: {}".format(mean_error / len(object_points)))

    # assert( np.all(dcoeffs == 0) )
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    params = (fx, fy, cx, cy)

    print()
    print('all units below measured in pixels:')
    print('  fx = {}'.format(K[0, 0]))
    print('  fy = {}'.format(K[1, 1]))
    print('  cx = {}'.format(K[0, 2]))
    print('  cy = {}'.format(K[1, 2]))
    print()
    print('pastable into Python:')
    print('  fx, fy, cx, cy = {}'.format(repr(params)))
    print()
    print('json:')
    print('{')
    print('  "fx": {},'.format(K[0, 0]))
    print('  "fy": {},'.format(K[1, 1]))
    print('  "cx": {},'.format(K[0, 2]))
    print('  "cy": {},'.format(K[1, 2]))
    print('  "dist": {}'.format(dcoeffs.tolist()))
    print('}')

    cv2.destroyAllWindows()

    # Displaying required output
    print(" Camera matrix:")
    print( K )

    print("\n Distortion coefficient:")
    print(dcoeffs)

    # Save the calibration data
    np.save(camera_name + '/matrix.npy', K)
    np.save(camera_name + '/distortion.npy', dcoeffs)
    np.save(camera_name + '/rvecs.npy', rvecs)
    np.save(camera_name + '/tvecs.npy', tvecs)


if __name__ == '__main__':
    main()