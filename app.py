import gradio as gr
import os
import subprocess
from typing import Iterator, TextIO
import textwrap
import whisper
from whisper.utils import get_writer

model = whisper.load_model('tiny')
title = 'Add Captions(CC) to your videos'

def process_text(text: str, maxLineWidth = None):
    if maxLineWidth is None or maxLineWidth < 0:
        return text
    
    lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
    return '\n'.join(lines)

def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"

def write_vtt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    print('WEBVTT\n', file=file)
    for segment in transcript:
        text = process_text(segment['text'], maxLineWidth).replace('-->', '->')

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )


def convert_mp4_mp3(file, output="mp3"):
    """
    Convert the Input Video files to Audio files (MP4 -> MP3)
    using FFMPEG 
    """
    filename, ext = os.path.splitext(file)
    subprocess.call(['ffmpeg', '-y', '-i', file, f'{filename}.{output}'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    return f"{filename}.{output}"

def transcribe(video):
    """
    Transcribe the text in the video file using Whisper model
    and write the transcribed captions to the video
    """
    audio_file = convert_mp4_mp3(video)
    # CFG
    options = dict(beam_size=5, best_of=5, fp16=False)
    translate_options = dict(task='translate', **options)
    result = model.transcribe(audio_file, **translate_options)

    output_dir = ''
    audio_path = audio_file.split('.')[0]

    # Write Subtitle onto a .vtt file
    with open(os.path.join(output_dir, audio_path + '.vtt'), 'w') as f:
        # WriteVTT.write_result(result=result['segments'], file=f)
        # writer = get_writer('vtt', output_dir)
        # writer(result['segments'], f'{audio_path}.vtt')
        write_vtt(transcript=result['segments'], file=f)

    # Write the subtitles on the input video
    subtitle = audio_path + '.vtt'
    output_video = audio_path + '_subtitled.mp4'
    os.system(f'ffmpeg -i {video} -vf subtitles={subtitle} {output_video}')

    return output_video

block = gr.Blocks()
with block:
    with gr.Group():
        with gr.Box():
            with gr.Row().style():
                input_video = gr.Video(
                    label="Input Video",
                    type="filepath",
                    mirror_webcam=False
                )
                output_video = gr.Video()
            btn = gr.Button('Generate Subtitle Video')

        btn.click(transcribe, inputs=[input_video], outputs=[output_video])

block.launch(enable_queue=True)