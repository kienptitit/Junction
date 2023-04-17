import cv2
import os
import glob


def get_frame(video_paths, out_folder):
    videos = glob.glob(os.path.join(video_paths, '*.mp4'))
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cnt = 0
        while cap.isOpened:
            ret, frame = cap.read()
            if ret:
                if cnt % 30 == 0:
                    out_folder_frame = os.path.join(out_folder, str(cnt))
                    if not os.path.exists(os.path.join(out_folder, str(cnt))):
                        os.mkdir(out_folder_frame)
                    cv2.imwrite(os.path.join(out_folder_frame,
                                             os.path.basename(video_path).split('.')[0] + '_frame_' + str(
                                                 cnt) + '.jpg'),
                                frame)
            else:
                break
            cnt += 1
        print("End")


# get_frame(r"E:\data_juntion\192_168_5_103.mp4", r"E:\data_juntion\192_168_5_104.mp4",
#          r"E:\juntion2023_2\SuperGluePretrainedNetwork\assets\Anh3_4")

get_frame(r"E:\data_juntion", "out_frame")
