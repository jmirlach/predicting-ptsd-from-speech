import argparse
import audiofile as af
import os
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert Trauma dataset.')
    parser.add_argument('--root', default='../data/MAUS/maus_download', required=False)
    parser.add_argument('--target', default='../data/data/20221121_Sprachaufnahmen_Preprocessed', required=False)
    args = parser.parse_args()

    phrases_NuS = [
        "einst stritten sich nordwind und sonne",
        "wer von ihnen beiden wohl der stärkere wäre",
        "als ein wanderer",
        "der in einen warmen mantel gehüllt war",
        "des weges daherkam",
        "sie wurden einig",
        "dass derjenige für den stärkeren gelten sollte",
        "der den wanderer zwingen würde",
        "seinen mantel abzunehmen",
        "der nordwind blies mit aller macht",
        "aber je mehr er blies",
        "desto fester hüllte sich der wanderer",
        "in seinen mantel ein",
        "endlich gab der nordwind den kampf auf",
        "nun erwärmte die sonne die luft",
        "mit ihren freundlichen strahlen",
        "und schon nach wenigen augenblicken",
        "zog der wanderer seinen mantel aus",
        "da musste der nordwind zugeben",
        "dass die sonne von ihnen beiden der stärkere war"
    ]

    phrases_DtS = [
        "das schneiderlein zog weiter",
        "immer seiner spitzen nase nach",
        "nachdem es lange gewandert war",
        "kam es in den hof eines königlichen palastes",
        "und da es müdigkeit empfand",
        "so legte es sich ins gras und schlief ein",
        "während es da lag kamen die leute",
        "betrachteten es von allen seiten",
        "und lasen auf dem gürtel",
        "sieben auf einen streich",
        "ach sprachen sie",
        "was will der große kriegsherr hier mitten im frieden",
        "das muss ein mächtiger herr sein",
        "sie gingen und meldeten es dem könig",
        "und meinten wenn krieg ausbrechen sollte",
        "wäre das ein wichtiger und nützlicher mann",
        "dem man um keinen preise fortlassen dürfte"
    ]


    phrase_level_data = {}
    phrase_level_data['file'] = []
    phrase_level_data['start'] = []
    phrase_level_data['end'] = []
    phrase_level_data['subject'] = []
    phrase_level_data['text'] = []
    phrase_level_data['phrase'] = []
    phrase_level_data['time'] = []
    phrase_level_data['trauma'] = []

    files = os.listdir(args.root)
    for file in files:
        df = pd.read_csv(os.path.join(args.root,file),delimiter=';')
        if 'NuS' in file:
            phrases = phrases_NuS
        else:
            phrases = phrases_DtS
        last_end = -1
        for phrase in phrases:
            start_word = phrase.split(' ')[0]
            end_word = phrase.split(' ')[-1]
            df_tmp = df[df['BEGIN']>last_end]
            start_frame = df_tmp[df_tmp['ORT']==start_word]['BEGIN'].values[0]
            end_frame_row_token = df_tmp[df_tmp['ORT']==end_word]['TOKEN'].values[0]
            end_frame = df_tmp[df_tmp['TOKEN']==end_frame_row_token]['BEGIN'].values[-1] + df_tmp[df_tmp['TOKEN']==end_frame_row_token]['DURATION'].values[-1]
            last_end = end_frame
            start_time = float(start_frame)/48000
            end_time = float(end_frame)/48000
            phrase_level_data['file'].append(os.path.join(args.target,file[:-8],f"{file[-7:-4]}.wav"))
            phrase_level_data['start'].append(start_time)
            phrase_level_data['end'].append(end_time)
            phrase_level_data['subject'].append(file.split(' ')[0].split('_')[0])
            phrase_level_data['text'].append(file[-7:-4])
            phrase_level_data['phrase'].append(phrase)
            if 'vor' in file:
                phrase_level_data['time'].append('before')
            else:
                phrase_level_data['time'].append('after')
            if 'Px' in file:
                phrase_level_data['trauma'].append(1)
            else:
                phrase_level_data['trauma'].append(0)

    
    df_phrase_level = pd.DataFrame(phrase_level_data)
    df_phrase_level.to_csv('../data/MAUS/trauma_phrase_level.csv', index=False)
    print(df_phrase_level)
    