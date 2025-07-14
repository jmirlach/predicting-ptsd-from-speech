import pandas as pd
import os
import shutil

data_path = "../data/20221121_Sprachaufnahmen_Preprocessed"
output_path = "../data/MAUS/maus_upload"
transcripts_path = "../data/MAUS"

all_file_folders = os.listdir(data_path)
for file_folder in all_file_folders:
    print(file_folder)
    if 'Text1' in file_folder:
        if not (os.path.exists(os.path.join(data_path, file_folder, 'NuS.wav'))):
                print(f'skipped for {file_folder}')
                continue
        shutil.copy(os.path.join(data_path, file_folder, 'NuS.wav'), os.path.join(output_path, file_folder + '_NuS.wav'))
        shutil.copy(os.path.join(transcripts_path, 'nordwind_sonne_sentences.txt'), os.path.join(output_path, file_folder + '_NuS.txt'))
    elif 'Text2' in file_folder:
        if not (os.path.exists(os.path.join(data_path, file_folder, 'DtS.wav'))):
                print(f'skipped for {file_folder}')
                continue
        shutil.copy(os.path.join(data_path, file_folder, 'DtS.wav'), os.path.join(output_path, file_folder + '_DtS.wav'))
        shutil.copy(os.path.join(transcripts_path, 'tapfere_schneiderlein_sentences.txt'), os.path.join(output_path, file_folder + '_DtS.txt'))
