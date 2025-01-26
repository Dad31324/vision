Richmond Hill Robotics Team
FRC Team 9086

This repository has code framework for a vision processing system essential to autonomous mode.
The implementation is envisioned to run with three disparate cameras on a Raspberry Pi.

Technical Requirements:
   - Three cameras of disparate make, model, features and potentially frame rate
   - Producer Consumer template in python for true multiprocessing
   - Producer is a "metronome" timer app for camera synchronization
   - Messaging is offload to a separate thread to minimize latency in the real time loop
   - Data logging is offloaded to a separate thread to minimize latency in the real time loop
   - User programmable sleep functions to preserve CPU bandwidth in the real time loop
   - Interlocks to protect critical system resources: IO to the screen and file system
   - Roadblocks in "main" to ensure completion of all processes before cleanup
   - Frame rate processing in the real time loop targeting 50 mSec (100mSec fallback)
   - Localization accuracy in the real time loop targeting 2" at 3 feet.
   - No memory leaks.  No accumulated lag.
   - Expected duration within 0.1 sec of expected duration after 2000 camera acquisitions

Software Components:
    - camera_port_scanner.py      Auto detect cameras and their usb ports
    - camera_calibration.py       Process a library of image files and write calibration data
    - fieldmap.json               JSON file with april tag IDs, field coordinates and unit normals
    - manager_v3.py               Fully functional vision code: acquisition, detection, localization, logging

Wish List:
    - Connect the IO to real time memory tables or network tables for automation
    - Implement a voting scheme to decide best real time lcoation from multiple cameras with multiple tags each
    - Optimize camera settings (resolution, localization parameters, rame rate)

Revision Log:
