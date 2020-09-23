import numpy as np
import os
import music21 as m21
from data import pinyin
phonemes_table = []
phonemes_position = []
lyrics_table = []
lyrics_position = []

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
                token = event.lyrics[1].text+'3'
                token = pinyin.split_pinyin(token)
                if token[0] != '':              
                    lines.append([duration, midi, 0, token[0]])
                    lines.append([duration, midi, 1, token[1]])
                elif token[1] != '':
                    lines.append([duration, midi, 2, token[1]])
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
            if j < 15:
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
                durationOutput.append([temp, startTime, endTime])
                if temp not in phonemes_table:
                    phonemes_table.append(temp)
                    phonemes_position.append([temp, startTime, endTime, metadata_filename])
    return durationOutput

def get_information(filename):
    phonemes = []
    with open(filename, encoding='utf-8') as f:
        j = 0
        for line in f:
            if j < 1:
                j = j+1
                continue
            line = line.split('\n')[0].split('\t')
            if len(phonemes) == 0 and line[0] == 'sp':
                j += 1
            else:
                phonemes.append(line)
    return phonemes

def get_all_information():
    path = 'C:/Users/weiyayou/Desktop/Code/interval_segments'
    song = []
    songs = []
    song_id = 1
    index_id = 0
    
    while song_id < 11:
        if song_id < 10:
            filename = '00' + str(song_id) + '_' + str(index_id) + '.lab'
        else:
            filename = '0' + str(song_id) + '_' + str(index_id) + '.lab'
        if os.path.exists(os.path.join(path, filename)):
            index_id += 1
            sentence = get_information(os.path.join(path, filename))
            song.append(sentence)
        else:
            song_id += 1
            index_id = 0
            songs.append(song)
            song = []
    return songs

def get_interval():
    path = 'C:/Users/10569/Desktop/Code/ByteSing-pytorch/data/DB-DM-001-F-001/Interval'
    songs = []
    for files in os.listdir(path):
        songs.append(get_phoneme_duration(os.path.join(path, files)))
    return songs

def compare(songs, songs_interval, start_id):
    while start_id < 10:
        j = 0
        song = songs[start_id]
        song_interval = songs_interval[start_id-3]
        for sentence in song:
            for k in range(len(sentence)):
                if song_interval[j][0] == sentence[k][0] and song_interval[j][1] == float(sentence[k][1]) and song_interval[j][2] == float(sentence[k][2]):
                    j += 1
                else:
                    # print(i, k)
                    print(sentence[k])
                    print(song_interval[j], '\n')
                    j += 1
        start_id += 1

def main():
    # songs = get_all_information()
    songs_interval = get_interval()
    print(phonemes_table)
    # print(phonemes_position)
    phonemes_table.sort()
    print(phonemes_table)
    # compare(songs, songs_interval, 9)


if __name__ == '__main__':
	main()
