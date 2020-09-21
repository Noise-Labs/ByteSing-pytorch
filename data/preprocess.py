# coding:utf-
import argparse
import os
from multiprocessing import cpu_count
import wave
from pydub import AudioSegment
import music21 as m21
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from functools import partial
from tqdm import tqdm
import pinyin


def get_second_part_wave(wav, start_time, end_time):
    start_time = int(start_time * 1000)
    end_time = int(end_time * 1000)
    sentence = wav[start_time: end_time]
    return sentence

def get_music_score(metadata_filename):
    # 处理乐谱，输出每个音素[持续时长，midi，因素类型，音素]
    lines = []
    score = m21.converter.parse(metadata_filename)
    part = score.parts.flat
    for i in range(len(part.notesAndRests)):
        event = part.notesAndRests[i]
        if isinstance(event, m21.note.Note):
            duration = event.seconds
            midi = event.pitch.midi
            if len(event.lyrics) > 0:
                token = event.lyrics[1].text+'3'
                token = pinyin.split_pinyin(token)
                if token[0] != '':              
                    lines.append([duration, midi, 0, token[0]])
                    lines.append([duration, midi, 1, token[1]])
                elif token[1] != '':
                    lines.append([duration, midi, 2, token[1]])
            else:
                temp = lines[-1]
                lines[-1][0] = lines[-1][0] + duration
        elif isinstance(event, m21.note.Rest):
            duration = event.seconds
            midi = 0
            token = 'sp'
            if len(lines) == 0 or lines[-1][-1] != 'sp':
                lines.append([duration, midi, 2, token])
            else:
                lines[-1][0] = lines[-1][0] + duration
    return lines

def get_phoneme_duration(metadata_filename):
    # 处理音频时长标注信息，返回[开始时间，结束时间，对应音素]
    with open(metadata_filename, encoding='utf-8') as f:
        i = 0
        j = 0
        durationOutput = []
        for line in f:
            if j != 12:
                j = j+1
                continue
            line = line.split('\n')[0]
            if i == 0:
                startTime = float(line)
                i = i+1
            elif i == 1:
                endTime = float(line)
                i = i+1
            else:
                i = 0
                temp = line.split('"')[1]
                if temp == 'sil' or temp == 'pau' or temp == "rest" or temp == "***sp":
                    temp = 'sp'
                if j == 12 or durationOutput[-1][2] != temp:
                    durationOutput.append([startTime, endTime, temp])
                else:
                    durationOutput[-1][1] = endTime
    return durationOutput

def audio_process_utterance(wav_dir, duration_dir, score_dir, index, wav, durations, scores):
    # Write the spectrogram and audio to disk
    audio_filename = 'audio-{}.wav'.format(index)
    duration_filename = 'duration-{}.npy'.format(index)
    score_filename = 'score-{}.npy'.format(index)
    wav.export(os.path.join(wav_dir, audio_filename), format="wav")
    np.save(os.path.join(duration_dir, duration_filename), durations, allow_pickle=False)
    np.save(os.path.join(score_dir, score_filename), scores, allow_pickle=False)

    # Return a tuple describing this training example
    return (audio_filename, duration_filename, score_filename)

def build_from_path(input_dir, wav_dir, score_dir, duration_dir, n_jobs=12, tqdm=lambda x: x):
    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    score_path = os.path.join(input_dir, 'Musicxml')
    duration_path = os.path.join(input_dir, 'interval')
    voice_path = os.path.join(input_dir, 'Vox')
    slice_head = []

    for files in os.listdir(score_path):
        basename = files.split('.')[0]
        scores = get_music_score(os.path.join(score_path, files))
        durations = get_phoneme_duration(os.path.join(duration_path, '{}.interval'.format(basename)))
        song = AudioSegment.from_wav(os.path.join(voice_path, '{}.wav'.format(basename)))
        sentence_duration = []
        score_index = -1
        for i in range(len(scores)):
            sentence_duration.append(durations[i])
            if durations[i][2] == 'sp':
                sentence_score = []
                wav = get_second_part_wave(song, sentence_duration[0][0], sentence_duration[-1][1])
                while True:
                    if i == 0 and scores[0][3] != 'sp':
                        slice_head = [True, int(sentence_duration[-1][1]*1000)]
                        # song = song[int(sentence_duration[-1][1]*1000): int(durations[-1][1]*1000)]
                        sentence_duration = []
                        break
                    score_index += 1
                    sentence_score.append(scores[score_index])
                    if scores[score_index][3] == 'sp':
                        futures.append(executor.submit(partial(audio_process_utterance, wav_dir,\
                            duration_dir, score_dir, basename + '-' + str(index), wav, sentence_duration, sentence_score)))
                        index += 1
                        sentence_duration = []
                        break
    
    return [future.result() for future in tqdm(futures) if future.result() is not None]
    # return futures

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')

def main():
    print('initializing preprocessing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='C:/Users/10569/Desktop/Code/ByteSing-pytorch/data')
    parser.add_argument('--dataset', default='demo')
    parser.add_argument('--output', default='training_data')
    parser.add_argument('--n_jobs', type=int, default=cpu_count())
    args = parser.parse_args()
	
	# Prepare directories
    in_dir  = os.path.join(args.base_dir, args.dataset)
    out_dir = os.path.join(args.base_dir, args.output)
    wav_dir = os.path.join(out_dir, 'audio')
    dur_dir = os.path.join(out_dir, 'duration')
    sco_dir = os.path.join(out_dir, 'score')
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(dur_dir, exist_ok=True)
    os.makedirs(sco_dir, exist_ok=True)
	
	# Process dataset
    metadata = []
    metadata = build_from_path(in_dir, wav_dir, sco_dir, dur_dir, args.n_jobs, tqdm=tqdm)
	# Write metadata to 'train.txt' for training
    write_metadata(metadata, out_dir)


if __name__ == '__main__':
	main()

    

