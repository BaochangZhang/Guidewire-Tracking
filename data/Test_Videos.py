import glob
from Utilities.file_folder_operation import *
import shutil

if __name__ == '__main__':
    # maybe_makedirs('./sythetic_Testvideo_det')
    # Test_Path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/sythetic_detection/Test_data'
    # Test_files = glob.glob(join(Test_Path, "*/*_image.avi"))
    # for video in Test_files:
    #     shutil.copy(video, './sythetic_Testvideo_det')

    # maybe_makedirs('./raw_Testvideo_det')
    # Test_Path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/raw_detection/Test_data'
    # Test_files = glob.glob(join(Test_Path, "*/*_image.avi"))
    # for video in Test_files:
    #     shutil.copy(video, './raw_Testvideo_det')

    maybe_makedirs('./raw_Testimage_det1')
    Test_Path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/raw_detection/Test_data'
    Test_files = sorted(glob.glob(join(Test_Path, "*/Image/*.png")))

    for image_path in Test_files:
        squence_name = image_path.split('/')[-3]
        squence_path = join('./raw_Testimage_det1',squence_name)
        maybe_makedirs(squence_path)
        filename = squence_name+'_'+image_path.split('/')[-1]
        new_path = join(squence_path,filename)
        shutil.copyfile(image_path, new_path)
        # shutil.copy(video, './raw_Testimage_det')


