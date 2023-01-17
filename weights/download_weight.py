from Detection_phase.utils.google_utils import attempt_download

for x in ['s', 'm', 'l', 'x']:
    attempt_download(f'yolov5{x}.pt')