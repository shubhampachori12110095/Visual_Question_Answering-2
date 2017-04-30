import os
import sys
import subprocess
import json
from movieqa_importer import MovieQA

def associate_additional_QA_info(QAs):
        """Get some information about the questions like story index and correct option.
        """

        qinfo = []
        for QA in QAs:
            qinfo.append({'qid':QA.qid,
                          'movie':QA.imdb_key,
                          'video_clips':QA.video_clips})
        return qinfo

if __name__ == "__main__":
    mqa = MovieQA.DataLoader()
    stories, QAs = mqa.get_story_qa_data('full', 'split_plot')
    qinfo = associate_additional_QA_info(QAs)
    with open('train_split.json') as fid:
        trdev = json.load(fid)
    qinfo = [qi for k, qi in enumerate(qinfo) if (qi['movie'] in trdev['train'] or qi['movie'] in trdev['dev'] or qi['qid'].startswith('val'))]
    frame_dir = '/home/jwk/Documents/MovieQA_benchmark/story/video_frames/'
    video_dir = '/home/jwk/Documents/MovieQA_benchmark/story/video_clips/'
    for info in qinfo:
        dir_name = frame_dir + info['qid'] + '/'
        clips = info['video_clips']
        key = info['movie']
        if len(clips) == 0:
            continue
        else:
            os.mkdir(dir_name)
            clip_dir = video_dir + key + '/'
            clips = list(set(clips))
            clips.sort()
            for i, c in enumerate(clips):
                command = "ffmpeg -i " + clip_dir + c + " -r 1 -s 224x224 -f image2 " + dir_name + "clips_" + str(i) + "_frame_%03d.jpg"
                out = subprocess.call(command, shell=True)
