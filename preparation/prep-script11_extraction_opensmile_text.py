import pandas as pd
import opensmile
import os
import audiosegment
from pydub import AudioSegment
import soundfile as sf
import tqdm


############################## SPECIFY PARAMETERS FIRST ##############################
FEATURE_SET = 'eGeMAPSv02' # Feature set that should be used: eGeMAPSv02 or ComParE_2016
FEATURE_LEVEL = 'Functionals' # Feature level that should be extracted: Functionals or LLD
AUDIO_PATH = '../data/cropped_data/cropped_text_data/per_word' # Path in which all raw audio files are located 
OUTPUT_PATH = '../data/features/features_text' # Path in which the generated csv-files are stored
IS_STEREO = True # If it is a stereo audio file, it will be transformed to mono audio in a pre-processing step.
######################################################################################


def get_audio_file_duration(audio_file):
    f = sf.SoundFile(audio_file)
    #print('samples = {}'.format(f.frames))
    #print('sample rate = {}'.format(f.samplerate))
    duration_in_ms = f.frames / f.samplerate * 1000
    return duration_in_ms


def convert_mp3_to_wav(input_path, file_name, output_dir):
    output = os.path.join(output_dir, "{}.wav".format(file_name[:-4]))
    sound = AudioSegment.from_mp3(os.path.join(input_path, file_name))
    sound.export(output, format="wav")


def convert_stereo_to_mono(file_name, output_dir):
    sound = AudioSegment.from_wav(file_name)
    sound = sound.set_channels(1)
    sound.export(output_dir, format="wav")


def extract_audio_file_features(audio, feature_set, feature_level):
    if feature_set=='eGeMAPSv02':
        if feature_level=='Functionals':
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        elif feature_level=='LLD':
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
    elif feature_set=='ComParE_2016':
        if feature_level=='Functionals':
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        elif feature_level=='LLD':
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )

    y = smile.process_file(audio)
    return pd.DataFrame(y)


def chunk_audio_file_and_extract_features(tmp_path, audio_file, hop_size, win_length, feature_set, feature_level):
    audio_file_duration = get_audio_file_duration(audio_file=audio_file)
    start = 0
    features_list = []
    while start+win_length < audio_file_duration:
        audio_seg = audiosegment.from_file(audio_file)
        chunked_audio = audio_seg[start:start+win_length]
        chunked_audio.export(os.path.join(tmp_path, 'tmp.wav'), "wav")
        feature_df = extract_audio_file_features(os.path.join(tmp_path, 'tmp.wav'), feature_set=feature_set, feature_level=feature_level)
        feature_df.insert(0,'file_path',audio_file)
        feature_df.insert(1,'start',start)
        feature_df.insert(2,'end',start+win_length)
        features_list.append(feature_df)
        os.system("rm {}".format(os.path.join(tmp_path, 'tmp.wav')))
        start += hop_size
    features_df = pd.concat(features_list, ignore_index=True)
    return features_df


def extract_opensmile_features(audio_path, original_filename, output_path, feature_set, start, end):
    tmp_path = os.path.join(output_path,f"{audio_path.split('/')[-2]}_{int(start*1000)}_{int(end*1000)}.wav")
    audio_seg = audiosegment.from_file(audio_path)
    chunked_audio = audio_seg[int(start*1000):int(end*1000)]
    chunked_audio.export(tmp_path, "wav")
    feature_df = extract_audio_file_features(tmp_path, feature_set=feature_set, feature_level="Functionals")
    feature_df.insert(0,'file',original_filename)
    feature_df.insert(1,'start',start)
    feature_df.insert(2,'end',end)
    os.remove(tmp_path)
    return feature_df



if __name__=='__main__':
    output_path=OUTPUT_PATH

    df = pd.read_csv('../data/MAUS/trauma_word_level.csv')
    if IS_STEREO==True:
        for ind,row in df.iterrows():
            file = row['file']
            if not os.path.exists(file[:-4]+'_mono.wav'):
                convert_stereo_to_mono(file_name=file, output_dir=file[:-4]+'_mono.wav')

    features_list = []
    for ind, row in df.iterrows():
        if ind%10==0:
            print(f"{ind+1}/{len(df)}")
        original_file = row['file']
        file = original_file[:-4]+'_mono.wav'
        start = row['start']
        end = row['end']
        features = extract_opensmile_features(audio_path=file, original_filename=original_file, output_path=output_path, feature_set=FEATURE_SET, start=start, end=end)
        features_list.append(features)
    features_df = pd.concat(features_list)
    features_df.to_csv(os.path.join(output_path,"word_opensmile.csv"),index=False)
    print(features_df)
