import time
import multiprocessing as mp
import cv2
import pyapriltags as at
import numpy as np
import json as j
import csv

#TODO:  Convert from file based I/O to real time memory tables
#TODO:  Vote on a consensus position using multiple april tags in a frame
#TODO:  Check accuracy of localization in field coordinate system
#TODO:  Requirements document and programmers hints via README for maintenance
#TODO:  GitHub repository for vision


def timer(camera1_queue_in, camera2_queue_in, camera3_queue_in, messenger_queue_in, frame_rate, number_of_frames):
    # This is the producer in a classic producer-consumer template.
    # Producer function is sending a trigger to the cameras to
    # take a snapshot and look for april tags
    message_string = "Starting the producer loop for camera synchronization."
    messenger_queue_in.put(message_string)

    # Insert a time delay to make sure the cameras are initialized
    # Those initializations are happening elsewhere.
    time.sleep(5)
    message_string = "Starting the acquisition loop."
    messenger_queue_in.put(message_string)

    # Loop over the requested number of frames
    # By queing the cameras in this way we make sure to keep them
    # in synch and try to avoid drift over the course of the game.
    # Drift is still possible if the camera queues are not drained.
    # We will monitor for this during development as we choose an optimal frame rate.
    tic_start = time.perf_counter()
    for ii in range(number_of_frames):
        tic = time.perf_counter()
        camera1_queue_in.put(1)    #This triggers a frame from camera 1
        camera2_queue_in.put(1)    #This triggers a frame from camera 2
        camera3_queue_in.put(1)    #This triggers a frame from camera 3
        #message_string = "Trigger frame grab # " + str(ii) + ". Time = " + str(tic)
        #messenger_queue_in.put(message_string)
        time.sleep(frame_rate)

    # Check integrated time for effective latency
    tic_end = time.perf_counter()
    dt_expected = number_of_frames * frame_rate
    dt_actual   = tic_end - tic_start
    message_string = "Finished with acquisition"
    messenger_queue_in.put(message_string)
    message_string = "Expected acquisition time: " + str(dt_expected)
    messenger_queue_in.put(message_string)
    message_string = "Actual acquisition time: " + str(dt_actual)
    messenger_queue_in.put(message_string)

    # We have looped through all frame grabs.
    # Gracefullly shut down the consumer loops.
    # ----------------------------------------
    # first leave time to flush the queues
    message_string = "Flushing the consumer queues."
    messenger_queue_in.put(message_string)
    time.sleep(1)
    # send notice
    message_string = "Sending command to stop camera acquisition."
    messenger_queue_in.put(message_string)
    # send the abort signal. Consumer flushes the queue and moves to cleanup
    # Abort signals cascade.  at_detector -> localizer -> logger.
    # Messaging App is aborted last from below.
    message_string='999'
    camera1_queue_in.put(message_string)
    camera2_queue_in.put(message_string)
    camera3_queue_in.put(message_string)
    return 1

def at_detector(camera_queue_in, at_queue_in, logger_queue_in, messenger_queue_in, camera_name, camera_port, sleep_rate, tag_size, llock):
    # Consumer template.
    # One instance runs for each camera and each instance has its own queues
    # for the camera frame grabs and the localization.  These queues are initialized in
    # main and passed here as variables in the method call for each camera.  This code re-use
    # really simplifies debugging.  Same code.  Three separate instances are running.
    #
    # Tasks accomplished here include grabbing a frame, detecting if april tag(s)
    # are present in the frame, and if so sending the frame to the localizer method.
    # Input variables are as follows:
    # .... camera_queue_in receives a trigger from the producer loop. Grab a frame or exit.
    # .... at_queue_in sends frames from here to the localizer for processing.
    # .... messenger_queue_in processes text strings to screen for debug and timing.
    # .... camera name is a prefix for finding the camera calibration data.
    # .... camera_port is the usb port.  This can change from run to run.  Be careful.
    # .... rate is the rate to attempt grabbing images at.
    # Testing is required to make sure the frame rate is compatible with the processing
    # speed to avoid latency in position, but on this laptop even with heavy I/O for debugging
    # we can support frame rates of 50 mSec.  We will repeat this on a Raspberry Pi next.
    message_string = "..... " + camera_name + " : Starting the April Tag Detector"
    messenger_queue_in.put(message_string)

    # Create some parameters for aprilTag plotting (debug only)
    # --------------------------------------------
    LINE_LENGTH = 5
    CENTER_COLOR = (125, 0, 0)
    CORNER_COLOR = (125, 0, 0)
    plot_diagnostics = False

    # Initialize the camera.  This is slow.  Do it once only at startup.
    message_string = "..... " + camera_name + " : Starting camera on USB Port " + str(camera_port)
    messenger_queue_in.put(message_string)
    cam = cv2.VideoCapture(camera_port)

    message_string = "..... " + camera_name + " : Confirmed connection"
    messenger_queue_in.put(message_string)

    # Load the intrinsic camera matrix from calibration
    mat = np.load(camera_name + '/matrix.npy')

    # Load the distortion coefficients from calibration
    dist = np.load(camera_name + '/distortion.npy')

    # Prepare a tuple with intrinsic camera parameters
    camera_param = (float(mat[0,0]), float(mat[1,1]), float(mat[0,2]), float(mat[1,2]))
    message_string = "..... " + camera_name + " : Loaded calibration data"
    messenger_queue_in.put(message_string)

    # Initialize the April Tag detector class
    # These parameters can (should!) be optimized during development
    at_detector = at.Detector(searchpath=['apriltags'],
                            families='tag36h11',
                            nthreads=1,
                            quad_decimate=2.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

    # Capture camera images at regular intervals governed by the producer loop.
    # This ensures synch as many cameras as you like without accumulation of drift
    # during the game.
    keep_going = True
    while keep_going:
        # wait for a trigger from the watchdog timer
        trigger = camera_queue_in.get()
        # record the time the trigger is received.
        toc0 = time.perf_counter()
        #message_string = '..... ' + camera_name + ': Trigger received : ' + str(toc0)
        #messenger_queue_in.put(message_string)
        # process the trigger.
        if trigger == "999":
            # send a stop to the localizer
            # message_string = '..... ' + camera_name + ' : Sending stop to the localizer'
            # messenger_queue_in.put(message_string)
            at_queue_in.put("999")
            # Commented out to avoid a race condition.  Localizer should kill the logger.
            #message_string = '..... ' + camera_name + ' : Sending stop to the data logger'
            #messenger_queue_in.put(message_string)
            #logger_queue_in.put("999")
            # end this process
            keep_going = False
        else:
            # Take a frame from the camera.
            # Log the time of the capture
            # On this latop we note a time delay ~ 2 mSec between receipt of trigger
            # and completion of the frame grab.  Super fast even with USB expansion port.
            ret, frame = cam.read()
            toc1 = time.perf_counter()
            # message_string = '..... ' + camera_name + ': Grabbed a frame : ' + str(toc1)
            # messenger_queue_in.put(message_string)
            # If a frame is received then see if it has an april tag
            if ret:
                # Convert to gray scale
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame

                # remove distortion.  Retain all pixels to preserve intrinsic camera parameters
                ungray = cv2.undistort(gray, mat, dist, None, None)

                # april tag detector
                # Timing data teaches us that there is about a 7 msec time delay
                # between grabbing the frame, confirming it has a tag present,
                # converting to gray scale, removing distortion, and
                # sending that to the localizer.
                # 9 mSec total time budget from trigger to completion of the next step.
                # Should be very acceptable.  I don't think messaging is slowing us down
                # but we should remember to disable or delete those tasks for the game.
                tags = at_detector.detect(ungray, estimate_tag_pose=True, camera_params=camera_param, tag_size=tag_size)
                toc2 = time.perf_counter()
                # message_string = '..... ' + camera_name + ' : Finished april tag detection  (' + str(len(tags)) + ') : ' + str(toc2)
                # messenger_queue_in.put(message_string)

                if len(tags) > 0:
                    # send a valid tag to the localizer
                    at_queue_in.put(tags)
                    # send a string to the messenger method
                    toc3 = time.perf_counter()
                    # trigger_string = '..... ' + camera_name + ' : April Tag sent to the localization method: ' + str(toc3)
                    # messenger_queue_in.put(trigger_string)
                    if plot_diagnostics:
                        # This logic block was strictly for debugging during development.
                        # Keep it turned off unless we suspect something is broken.
                        corner_point_world = np.array([
                                      [0, 2*tag_size, 0],
                                      [ 2*tag_size, 2*tag_size, 0],
                                      [ 2*tag_size, 0, 0],
                                      [ 0,  0, 0]
                                      ])
                        ret, rvecs, tvecs = cv2.solvePnP(corner_point_world, tags[0].corners.astype(float), mat, dist)
                        image_2, _ = cv2.projectPoints(corner_point_world, rvecs, tvecs, mat, dist)
                        for kk in range(4):
                            corner_point = (round(image_2[kk][0,0].item()), round(image_2[kk][0,1].item()))
                            print(".......... Corner Point " + str(kk) + " : " + str(corner_point) + " : " + str(corner_point_world[kk]))
                            image = cv2.circle(ungray, corner_point, 2, CORNER_COLOR, 2)

                            # refresh the camera image
                            cv2.imshow('Result', image)
                            cv2.waitKey(5)
                else:
                    continue
                    # trigger_string = "....." + camera_name + " : No April Tag in the frame"
                    # messenger_queue_in.put(trigger_string)

            else:
                # This should not happen.
                trigger_string = "....." + camera_name + " : No frame returned.  Abort"
                at_queue_in.put("999")  # aborts the localizer loop
                messenger_queue_in.put("999") # aborts the messenger loop

            # Wait for user specified time interval.  Keep this very short unless testing.
            time.sleep(sleep_rate)

    # Send a stop signal to the other processes
    at_queue_in.put("999")            # aborts the localizer loop (redundant)
    # logger_queue_in.put("999")        # aborts the data logger
    # messenger_queue_in.put("999")     # aborts the watcher loop

    # If you get here then we have stopped looping over frames.
    # Time delay to let all the other queues get flushed.
    time.sleep(2)

    # Release the video capture object
    cam.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return 1


def localizer(at_queue_out, logger_queue_out, messenger_queue_in2, camera_name, sleep_rate, tag_size, llock):
    # Consumer function that localizes the camera from the april tag pose estimates
    #      - at_queue_out is a queue delivering april tags from the camera process to the detector process
    #      - messenger_queue_in2 sends strings to the messenger app
    message_string = "..... " + camera_name + " : Starting the localizer"
    messenger_queue_in2.put(message_string)

    # Load the calibration data (only needed for solvePnP debug)
    mat = np.load(camera_name +  '/matrix.npy')
    dist = np.load(camera_name + '/distortion.npy')

    # Define the world coordinates in the tag frame of reference
    xyz_world = np.array([tag_size, tag_size, 0, 1])

    # Define the aprilTag meta data in the field frame of reference
    with open('fieldmap.json', 'r') as file:
        field_map_data = j.load(file)
        #print(type(field_map_data))
        #print(field_map_data.get('tags')[2].get('xyz'))

    # Make a list of tag ids for lookup later in the loop
    tag_ids = [field_map_data['tags'][ii]['id'] for ii in range(0,len(field_map_data['tags']))]

    # Make a list of direction cosines for lookup later in the loop
    tag_theta = [np.atan2(field_map_data['tags'][ii]['unit_vector'][1],
                          field_map_data['tags'][ii]['unit_vector'][0])
                 for ii in range(0,len(field_map_data['tags']))]

    # Define a transformation matrix from april tag to robot
    T = np.array([
        [0,0,1],
        [1,0,0],
        [0,1,0]
        ])

    # print_diagnostics.  Set to false after dev testing.
    print_diagnostics = False

    # write results to csv.  Leave this true until we setup network tables.
    write_csv_flag = True

    keep_going = True
    while keep_going:
        # wait for a new trigger on the queue
        # triggers sent this way are tag metadata from april tag detection
        tags = at_queue_out.get()
        if tags == "999":
            # This is the signal to gracefully shut down.
            message_string = "..... " + camera_name + " : Telling the data logger to write to disk"
            messenger_queue_in2.put(message_string)
            logger_queue_out.put("999")
            keep_going = False
        elif len(tags) > 0:
            # get a time stamp for checking latency
            toc1 = time.perf_counter()
            # send a string to the messenger method
            # message_string = '..... ' + camera_name + ' : April Tag received by the localizer : ' + str(toc1)
            # messenger_queue_in2.put(message_string)

            # Loop over the number of tags that were found
            # messenger_queue_in2.put('..... ' + camera_name + ' :  Detections in this camera image: '  + str(len(tags)))
            for ii in range(len(tags)):
                # transform from camera pose to field pose
                RT = np.array([
                    [tags[ii].pose_R[0,0], tags[ii].pose_R[0,1], tags[ii].pose_R[0,2], tags[ii].pose_t[0,0]],
                    [tags[ii].pose_R[1,0], tags[ii].pose_R[1,1], tags[ii].pose_R[1,2], tags[ii].pose_t[1,0]],
                    [tags[ii].pose_R[2,0], tags[ii].pose_R[2,1], tags[ii].pose_R[2,2], tags[ii].pose_t[2,0]],
                    [0,0,0,1]
                    ])
                xyz_tag      = np.matmul(RT, xyz_world)
                xyz_camera   = np.matmul(T, [xyz_tag[0],xyz_tag[1],xyz_tag[2]])
                #####print(xyz_camera)

                # Find the index for this tag in the json field map
                try:
                    index = tag_ids.index(tags[ii].tag_id)
                    xyz_tag = field_map_data['tags'][index]['xyz']
                    theta = -1 * tag_theta[index]
                except ValueError:
                    theta = 0
                    xyz_tag = [0.0, 0.0, 0.0]
                #####print(xyz_tag)

                # Here is a rotation matrix to transform from the
                # tag coordinate system to the field coordinate system.
                # It is computed from the tag unit normal in the field map
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                    ])
                #####print('Rotation matrix : ' + str(np.shape(rotation_matrix)[0]) + ' : ' + str(np.shape(rotation_matrix)[1]))
                #####print('xyz_camera      : ' + str(np.shape(xyz_camera)) )


                # Perform the transformation to field coordinate system
                xy_camera = np.array([xyz_camera[0], xyz_camera[1]])
                xy_tag    = np.array([xyz_tag[0], xyz_tag[1]])
                xy_field  = xy_tag + np.matmul(xy_camera, rotation_matrix)
                xyz_field = np.array([xy_field[0], xy_field[1], xyz_tag[2]+xyz_camera[2]])

                corner_point_world = np.array([
                                      [0, 2*tag_size, 0],
                                      [ 2*tag_size, 2*tag_size, 0],
                                      [ 2*tag_size, 0, 0],
                                      [ 0,  0, 0]
                                      ])
                ret, rvecs, tvecs = cv2.solvePnP(corner_point_world, tags[ii].corners.astype(float), mat, dist)

                toc2 = time.perf_counter()
                # message_string = "..... " + camera_name + " : This localization is complete:   " + str(toc2)
                # messenger_queue_in2.put(message_string)

                #-------- DEBUGGING: get roll, pitch, yaw and translations from solvePnP
                if print_diagnostics:

                    message_string = ".........." + camera_name + ": Pitch (degrees):   " + str(rvecs[0]*180/3.14159)
                    messenger_queue_in2.put(message_string)
                    message_string = ".........." + camera_name + ": Yaw (degrees):   " + str(rvecs[1]*180/3.14159)
                    messenger_queue_in2.put(message_string)
                    message_string = ".........." + camera_name + ": Roll (degrees):   " + str(rvecs[2]*180/3.14159)
                    messenger_queue_in2.put(message_string)
                    message_string = ".........." + camera_name + ": dX (Inches):   " + str(xyz_camera[0]*1/0.0254)
                    messenger_queue_in2.put(message_string)
                    message_string = ".........." + camera_name + ": dY (Inches):   " + str(xyz_camera[1]*1/0.0254)
                    messenger_queue_in2.put(message_string)
                    message_string = ".........." + camera_name + ": dZ (Inches):   " + str(xyz_camera[2]*1/0.0254)
                    messenger_queue_in2.put(message_string)
                    toc4 = time.perf_counter()

                    message_string = "..... " + camera_name + " : This transformation is complete:   " + str(toc4)
                    messenger_queue_in2.put(message_string)

                    message_string = "------------------------------------"
                    messenger_queue_in2.put(message_string)

                # Write to the common csv file for post processing.
                # We will still need to parse these to select a "truth" position.
                # ['TAG ID' 'Roll', 'Pitch', 'Yaw', 'X','Y','Z']
                if write_csv_flag == True:
                    csv_string = [
                    str(ii),
                    camera_name,
                    str(toc1),
                    str(toc2),
                    str(tags[ii].tag_id),
                    str(rvecs[2]*180/3.14159), str(rvecs[0]*180/3.14159), str(rvecs[1]*180/3.14159),
                    str(xyz_camera[0]*1/0.0254), str(xyz_camera[1]*1/0.0254), str(xyz_camera[2]*1/0.0254),
                    str(xyz_field[0]*1/0.0254),  str(xyz_field[1]*1/0.0254),  str(xyz_field[2]*1/0.0254),
                    ]
                    logger_queue_out.put(csv_string)

                # sleep to unload the cpu
                time.sleep(sleep_rate)



def logger(logger_queue_in, camera_name,  messenger_queue_out, llock):
    # data logging app to minimize open and close overhead in loop
    # wipe any pre-existing copy
    csv_filename = camera_name + '.csv'
    with open(csv_filename, 'w', newline='') as fid:
            writer = csv.writer(fid)
    message_string = "..... " + camera_name + " : Initialized csv file"
    messenger_queue_out.put(message_string)

    # Accumulate data in memory for best performance
    my_data = []
    keep_looking = True
    while keep_looking:
        # wait for a new trigger on the queue
        new_message = logger_queue_in.get()
        if new_message == "999":
            message_string = "..... " + camera_name + " : Transitioning to write data"
            messenger_queue_out.put(message_string)
            keep_looking = False
        else:
            my_data.append(new_message)

    # Dump to disk
    with llock:
        try:
            message_string = "..... " + camera_name + " : Writing data to disk"
            messenger_queue_out.put(message_string)
            csv_filename = camera_name + '.csv'
            with open(csv_filename, 'a', newline='') as fid:
                writer = csv.writer(fid)
                for ii in range(len(my_data)):
                    writer.writerow(my_data[ii])
            message_string = "..... " + camera_name + " : Writing is complete"
            messenger_queue_out.put(message_string)
            fid.close
        except:
            message_string = "..... " + camera_name + " : Error writing to disk"
            messenger_queue_out.put(message_string)

    return 1

def messenger(messenger_queue_out, rate, llock):
    # publishing function for managing IO
    with llock:
        print("Starting the Messenger.")

    while True:
        # wait for a new trigger on the queue
        new_message = messenger_queue_out.get()
        if new_message == "999":
            break
        with llock:
            print(new_message)


if __name__ == "__main__":
    # Start the queues for vision processing on 3 cameras
    # plus associated messaging and data logging.
    # ------------------------------------------
    wd_queue = mp.Queue()           # watchdog timer
    camera1_queue = mp.Queue()      # april tag detection on camera1
    camera2_queue = mp.Queue()      # april tag detection on camera2
    camera3_queue = mp.Queue()      # april tag detection on camera3
    at1_queue = mp.Queue()          # localizer from images on camera1
    at2_queue = mp.Queue()          # localizer from images on camera2
    at3_queue = mp.Queue()          # localizer from images on camera3
    logger1_queue = mp.Queue()      # data logger for images on camera1
    logger2_queue = mp.Queue()      # data logger for images on camera2
    logger3_queue = mp.Queue()      # data logger for images on camera3
    messenger_queue = mp.Queue()    # messaging app
    lock_messenger = mp.Lock()
    lock_logger = mp.Lock()

    # Other constants
    # ------------------------------------------
    number_of_cameras = 2
    tag_size = 0.08573    # from outer perimeter to outer perimeter divided by two
    rate1 = 0.050         # clock rate for taking images
    rate2 = 0.005         # sleeper for april tag detection
    rate3 = 0.005         # sleeper for localizer
    duration = 5          # vision processing duration in seconds
    frames = 10           # number of camera frames to process at rate1.

    # Initialize the producer loop which keeps all cameras in synch
    # --------------------------------------------
    producers  = [mp.Process(target=timer, args=(camera1_queue, camera2_queue, camera3_queue, messenger_queue,rate1, frames)) for _ in range(1)]
    for p in producers:
        p.start()

    # Start the first camera
    # ---------------------------------------------
    camera1_name = 'camera1'
    camera1_port = 2
    detector1  = [mp.Process(target=at_detector, args=(camera1_queue, at1_queue, logger1_queue, messenger_queue, camera1_name, camera1_port, rate2, tag_size, lock_messenger)) for _ in range(1)]
    localizer1 = [mp.Process(target=localizer, args=(at1_queue, logger1_queue, messenger_queue, camera1_name, rate3, tag_size, lock_messenger), daemon=True) for _ in range(1)]
    logger1    = [mp.Process(target=logger, args=(logger1_queue,camera1_name, messenger_queue, lock_logger), daemon=False) for _ in range(1)]
    for c1 in detector1:
        c1.start()
    for c2 in localizer1:
            c2.start()
    for c3 in logger1:
            c3.start()

    # Start the second camera
    # ----------------------------------------------
    if number_of_cameras >= 2:
        camera2_name = 'camera2'
        camera2_port = 0
        detector2  = [mp.Process(target=at_detector, args=(camera2_queue, at2_queue, logger2_queue, messenger_queue, camera2_name, camera2_port, rate2, tag_size, lock_messenger)) for _ in range(1)]
        localizer2 = [mp.Process(target=localizer, args=(at2_queue, logger2_queue, messenger_queue, camera2_name, rate3, tag_size, lock_messenger), daemon=True) for _ in range(1)]
        logger2    = [mp.Process(target=logger, args=(logger2_queue, camera2_name, messenger_queue, lock_logger), daemon=False) for _ in range(1)]
        for c4 in detector2:
            c4.start()
        for c5 in localizer2:
            c5.start()
        for c6 in logger2:
            c6.start()

    # Start the third camera
    # ----------------------------------------------
    if number_of_cameras >= 3:
        camera3_name = 'camera3'
        camera3_port = 6
        detector3= [mp.Process(target=at_detector, args=(camera3_queue, at3_queue, logger3_queue, messenger_queue, camera3_name, camera3_port, rate2, tag_size, lock_messenger)) for _ in range(1)]
        localizer3 = [mp.Process(target=localizer, args=(at3_queue, logger3_queue, messenger_queue, camera3_name, rate3, tag_size, lock_messenger), daemon=True) for _ in range(1)]
        logger3 = [mp.Process(target=logger, args=(logger3_queue, camera3_name, messenger_queue, lock_logger), daemon=False) for _ in range(1)]
        for c7 in detector3:
            c7.start()
        for c8 in localizer3:
            c8.start()
        for c9 in logger3:
            c9.start()

    # Initialize a messaging app
    # ----------------------------------------------
    watchers = [mp.Process(target=messenger, args=(messenger_queue, rate3, lock_messenger), daemon=True) for _ in range(1)]

    for w in watchers:
        w.start()

    for p in producers:
        p.join()

    # Use roadblocks to wait for the data loggers to finish writing
    message_string = "Waiting for all loggers to finish their write to file."
    messenger_queue.put(message_string)
    for id in logger1:
        id.join()

    if number_of_cameras >= 2:
        for id in logger2:
            id.join()

    if number_of_cameras >= 3:
        for id in logger3:
            id.join()

    # Stop the messaging app.  This could be tied to a loop until the logger apps are done
    message_string = "All done. Stopping the messaging app"
    messenger_queue.put(message_string)
    messenger_queue.put("999")
