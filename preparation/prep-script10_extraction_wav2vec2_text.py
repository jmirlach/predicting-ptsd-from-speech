import argparse
import audiofile
import audtorch
import json
import glob
import os
import pandas as pd
import tqdm
import torch
import torchaudio

from transformers import (
    Wav2Vec2Model, 
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract w2v2 features')
    parser.add_argument('--src',default='../data/cropped_data/cropped_text_data/per_phrase')
    parser.add_argument('--dst',default='../data/features/features_text/phrase_facebook-wav2vec2.csv')
    parser.add_argument(
        '--model', 
        default='facebook/wav2vec2-large-xlsr-53-german',
        choices=[
            'facebook/wav2vec2-large-xlsr-53',
            'facebook/wav2vec2-base',
            'facebook/hubert-base-ls960',
            'facebook/wav2vec2-large',
            'facebook/hubert-large-ll60k',
            'facebook/wav2vec2-large-robust',
            'facebook/wav2vec2-large-100k-voxpopuli',
            'facebook/wav2vec2-xls-r-300m',
            'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
            'facebook/wav2vec2-large-xlsr-53-german'
        ]
    )
    parser.add_argument(
        '--device', 
        default= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    args = parser.parse_args()

    dst = args.dst
    if os.path.isfile(args.dst):
        exit()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    files = glob.glob(os.path.join(args.src, '*.wav'))

    vocab_dict = {}
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    tokenizer = Wav2Vec2CTCTokenizer('./vocab.json')
    tokenizer.save_pretrained('./tokenizer')

    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    except OSError:
        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, 
            sampling_rate=16000, 
            padding_value=0.0, 
            do_normalize=True, 
            return_attention_mask=True
        )
    processor = Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer)
    model = Wav2Vec2Model.from_pretrained(args.model).to(args.device)
    model.eval()
    
    num_features = 768 if 'base' in args.model else 1024
    embeddings = torch.zeros(len(files), num_features)
    for counter, (file) in tqdm.tqdm(
        enumerate(files), 
        total=len(files), 
        desc=args.model
    ):
        audio, fs = audiofile.read(
            file,
            always_2d=True
        )
        audio = audtorch.transforms.Expand(4000)(audio)
        audio = torch.from_numpy(audio)
        if fs != 16000:
            audio = torchaudio.transforms.Resample(fs, 16000)(audio)
        if len(audio.shape) == 2:
            audio = audio.mean(0)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings[counter, :] = model(
                inputs.input_values.to(args.device),
            )[0].cpu().mean(1).squeeze(0)
    features = pd.DataFrame(
        data=embeddings.numpy(),
        columns=[f'Neuron_{x}' for x in range(num_features)],
        index=pd.Index(files, name='file')
    ).reset_index()
    features['file'] = features['file'].apply(os.path.basename)
    features.to_csv(args.dst, index=False)



