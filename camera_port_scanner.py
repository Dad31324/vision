import cv2
from argparse import ArgumentParser

def main():
    # Create a parser for optional parameters
    parser = ArgumentParser(prog='Camera Port Scanner',
                            description='Camera Port Scannera')
    parser.add_argument('--min', type=int, default=0,  metavar='', help='Min port to scan')
    parser.add_argument('--max', type=int, default=10, metavar='', help='Max port to scan')

    args = parser.parse_args()
    min_port = args.min
    max_port = args.max

    if (min_port >= max_port):
        raise ValueError('Make sure the minimum port is less than the maximum!')

    working_ids = []
    for i in range(min_port, max_port):
        try:
            cam = cv2.VideoCapture(i)
        except:
            continue

        if cam.isOpened():
            working_ids.append(i)
            my_str = "USB Port " + str(i)
            ret, frame = cam.read()
            cv2.namedWindow(my_str, cv2.WINDOW_KEEPRATIO)
            cv2.imshow(my_str, frame)
            cv2.resizeWindow(my_str, 640, 400)
            cv2.waitKey(5000)

        cam.release()

    print('Working Camera Ports: \n \
    {}'.format(working_ids))

if __name__ == '__main__':
    main()