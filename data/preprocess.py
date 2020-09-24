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
import english_phoneme

def compare_score_duration(duration, score):
    print("duration的长度为：", len(duration))
    print("score的长度为：", len(score))
    length = len(score) if len(score)+1 < len(duration) else len(duration)
    for i in range(length):
        if duration[i][2] != score[i][3]:
            print("第一个不同出在：", i)
            break
    sp_score_count = 0
    for item in score:
        if item[3] == 'sp': sp_score_count += 1
    sp_duration_count = 0
    for item in duration:
        if item[2] == 'sp': sp_duration_count += 1
    if sp_score_count != sp_duration_count:
        print(score[0], duration[0])
        print(score[-1], duration[-1])
        print("score中的sp有：", sp_score_count)
        print("duration中的sp有：", sp_duration_count, '\n')
    
def slice_head_tail(duration, score, song):
    flag = False
    sliced_duration = []
    sliced_score = []
    head = 0
    tail = len(duration)
    if duration[0][2] == 'sp':
        flag = True
        song = song[int(duration[0][1]*1000):]
        head = 1
    if duration[-1][2] == 'sp':
        song = song[:int(duration[-1][0]*1000)]
        tail = len(duration) - 1
    sliced_duration = duration[head:tail]
    head = 0
    tail = len(score)
    if score[0][3] == 'sp':  head = 1
    if score[-1][3] == 'sp': tail = len(score) - 1
    sliced_score = score[head: tail]
    return (flag, sliced_duration, sliced_score, song)       

def get_second_part_wave(wav, start_time, end_time):
    start_time = int(start_time * 1000)
    end_time = int(end_time * 1000)
    sentence = wav[start_time: end_time]
    return sentence

# 处理乐谱，输出每个音素[持续时长，midi，因素类型，音素]
def get_music_score(metadata_filename):
    lines = []
    score = m21.converter.parse(metadata_filename)
    part = score.parts.flat
    for i in range(len(part.notesAndRests)):
        event = part.notesAndRests[i]
        if isinstance(event, m21.note.Note):
            duration = event.seconds
            midi = event.pitch.midi
            if len(event.lyrics) > 0:
                # 中文
                if event.lyrics[1].text.islower():
                    token = event.lyrics[1].text+'3'
                    token = pinyin.split_pinyin(token)
                    if token[0] != '':              
                        lines.append([duration, midi, 0, token[0]])
                        lines.append([duration, midi, 1, token[1]])
                    elif token[1] != '':
                        lines.append([duration, midi, 2, token[1]])
                # 英文，设英文开头为声母，其后都为韵母
                else:
                    token = english_phoneme.split_to_phoneme(event.lyrics[1].text)
                    lines.append([duration, midi, 0, token[0]])
                    phoneme_index = 1
                    while phoneme_index < len(token):
                        lines.append([duration, midi, 1, token[phoneme_index]])
                        phoneme_index += 1            
            else:
                # 乐谱中对多音符唱词后几个音符不会标注歌词，故采用将时长合入韵母的处理方式
                lines[-1][0] += duration
        elif isinstance(event, m21.note.Rest):
            duration = event.seconds
            midi = 0
            token = 'sp'
            if len(lines) == 0 or lines[-1][-1] != 'sp':
                lines.append([duration, midi, 1, token])
            else:
                lines[-1][0] = lines[-1][0] + duration
    return lines

# 处理音频时长标注信息，返回[开始时间，结束时间，对应音素]
def get_phoneme_duration(metadata_filename):    
    with open(metadata_filename, encoding='utf-8') as f:
        i = 0
        j = 0
        durationOutput = []
        for line in f:
            if j < 12:
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
                temp = temp.replace(" ", "")
                if temp == 'sil':
                    temp = 'sp'
                durationOutput.append([startTime, endTime, temp])
                # if j == 12:
                #     durationOutput.append([startTime, endTime, temp])
                #     j += 1
                # else:
                #     if durationOutput[-1][2] != temp:
                #         durationOutput.append([startTime, endTime, temp])
                #     else:
                #         durationOutput[-1][1] = endTime
    return durationOutput

def audio_process_utterance(wav_dir, duration_dir, score_dir, index, wav, durations, scores):
    start_time = durations[0][0]
    for i in range(len(durations)):
        durations[i][0] -= start_time
        durations[i][1] -= start_time
    
    if len(durations) == len(scores):
        # 假设当长度相同时，音素相同，只是表示方法不同，统一成pinyin转换成的音素表示法
        for i in range(len(durations)): 
            # durations[i][2] = scores[i][3]
            if durations[i][2] != scores[i][3]:
                print(index, "len相同，但音素不同")
    elif len(durations) > len(scores):
        j = 0
        i = 0
        # print(index, "len(durations) > len(scores)")
        while i < len(scores):
            # 假设音素不同时只有一种情况，长音音素后面标注和前不同，例如:interval中'uan'后长音标记为了'an'
            if scores[i][3] != durations[j][2]:
                if j > 0 : durations[j-1][1] = durations[j][1]
                del durations[j]
            else: 
                j += 1
                i += 1
        if i < len(durations):
            durations[j-1][1] = durations[-1][1]
            durations = durations[:j]
    else:
        j = 0
        print(index, 'len(durations) < len(scores)')
        # 假设 len（scores）>len（duraionts）只有一种情况：乐谱中没有标注句子间的休止符
        # 这个破坏了分句，需要在外部处理，，，暂时还没想到处理方法（期待数据集更新后没有这个问题QAQ）
    if len(durations) != len(scores):
        print("处理完后还是长度不同：", index)
    # Write the audio to disk
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

    score_path = os.path.join(input_dir, 'Musicxml')
    duration_path = os.path.join(input_dir, 'interval')
    voice_path = os.path.join(input_dir, 'Vox')

    # 循环处理每首歌曲
    for files in os.listdir(score_path):
        index = 1 # 用于记录对一首歌的切分号，便于存储
        basename = files.split('.')[0]
        scores = get_music_score(os.path.join(score_path, files)) # 每个音素[持续时长，midi，因素类型，音素]        
        durations = get_phoneme_duration(os.path.join(duration_path, '{}.interval'.format(basename)))

        # print(files)
        # compare_score_duration(durations, scores)

        song = AudioSegment.from_wav(os.path.join(voice_path, '{}.wav'.format(basename)))
        sliced_head, durations, scores, song = slice_head_tail(durations, scores, song)
        # song.export(os.path.join("C:/Users/10569/Desktop/Code/ByteSing-pytorch/data/training_data", "temp.wav"), format="wav")
        sentence_duration = [] # 存储当前句子的各个音素duration信息（以sp为划分标准）
        score_index = -1 # 存储当前句子的乐谱音素查找下标
        # silence_head = [False, 0]
        
        # 根据乐谱中获取的信息，循环处理各音素
        for i in range(len(durations)):
            sentence_duration.append(durations[i])
            if durations[i][2] == 'sp' or i == len(durations) - 1:
                sentence_score = []
                if sliced_head:
                    wav = get_second_part_wave(song, sentence_duration[0][0] - durations[0][0], sentence_duration[-1][1] - durations[0][0])
                else:
                    wav = get_second_part_wave(song, sentence_duration[0][0], sentence_duration[-1][1])
                while True:
                    score_index += 1
                    sentence_score.append(scores[score_index])
                    if scores[score_index][3] == 'sp'  or score_index == len(scores) - 1:
                        # futures.append(executor.submit(partial(audio_process_utterance, wav_dir,\
                        #     duration_dir, score_dir, basename + '-' + str(index), wav, sentence_duration, sentence_score)))
                        futures.append(audio_process_utterance(wav_dir, duration_dir, score_dir, basename + '-' + str(index), wav, sentence_duration, sentence_score))
                        index += 1
                        sentence_duration = []
                        break
            
    # return [future.result() for future in tqdm(futures) if future.result() is not None]
    return futures
    # return 0

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')

def main():
    print('initializing preprocessing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='C:/Users/10569/Desktop/Code/ByteSing-pytorch/data')
    parser.add_argument('--dataset', default='DB-DM-001-F-001')
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

    

