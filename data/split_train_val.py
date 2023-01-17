import glob
from Utilities.file_folder_operation import *
from Utilities.split_dataset import split


if __name__ == '__main__':
    # sythetic
    sythetic_Test_Path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/sythetic_detection_flow/Test_data'
    sythetic_Train_Path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/sythetic_detection_flow/Training_data'
    sythetic_Train_files = glob.glob(join(sythetic_Train_Path, "*/Image/*.png"))
    sythetic_Test_files = glob.glob(join(sythetic_Test_Path, "*/Image/*.png"))
    # sythetic_Train_files, sythetic_Val_files = split(sythetic_Train_files, shuffle=True, ratio=0.9)

    # maybe_makedirs('./sythetic_data/')
    # sythetic_txt_test = open('./sythetic_data/sythetic_test_guidewire_v2.txt', 'w')
    # sythetic_txt_train = open('./sythetic_data/sythetic_train_guidewire_v2.txt', 'w')
    # sythetic_txt_val = open('./sythetic_data/sythetic_val_guidewire.txt', 'w')
    # raw
    raw_Test_Path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/raw_detection_flow/Test_data'
    raw_Train_Path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/raw_detection_flow/Training_data'
    raw_Train_files = glob.glob(join(raw_Train_Path, "*/Image/*.png"))
    raw_Test_files = glob.glob(join(raw_Test_Path, "*/Image/*.png"))
    # raw_Train_files, raw_Val_files = split(raw_Train_files, shuffle=True, ratio=0.9)
    maybe_makedirs('./raw_data/')
    raw_txt_test = open('./raw_data/raw_test_guidewire_v2_flow.txt', 'w')
    raw_txt_train = open('./raw_data/raw_train_guidewire_v2_flow.txt', 'w')
    # raw_txt_val = open('./raw_data/raw_val_guidewire.txt', 'w')
    # sythetic+raw
    Test_files = raw_Test_files+sythetic_Test_files
    Train_files = raw_Train_files + sythetic_Train_files
    # Val_files = raw_Val_files + sythetic_Val_files
    maybe_makedirs('./all_data/')
    txt_test = open('./all_data/All_test_guidewire_v2_flow.txt', 'w')
    txt_train = open('./all_data/All_train_guidewire_v2_flow.txt', 'w')
    # txt_val = open('./all_data/All_val_guidewire.txt', 'w')

    # save sythetic file
    # for file in sythetic_Train_files:
    #     sythetic_txt_train.write(file + '\n')
    # for file in sythetic_Test_files:
    #     sythetic_txt_test.write(file + '\n')
    # # for file in sythetic_Val_files:
    # #     sythetic_txt_val.write(file + '\n')
    # sythetic_txt_train.close()
    # sythetic_txt_test.close()
    # # sythetic_txt_val.close()
    # save raw file
    for file in raw_Train_files:
        raw_txt_train.write(file + '\n')
    for file in raw_Test_files:
        raw_txt_test.write(file + '\n')
    # # for file in raw_Val_files:
    # #     raw_txt_val.write(file + '\n')
    raw_txt_train.close()
    raw_txt_test.close()
    # raw_txt_val.close()
    # # save raw+sythetic file
    # for file in Train_files:
    #     txt_train.write(file + '\n')
    # for file in Test_files:
    #     txt_test.write(file + '\n')
    # # for file in Val_files:
    # #     txt_val.write(file + '\n')
    # txt_train.close()
    # txt_test.close()
    # # txt_val.close()

