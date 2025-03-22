
import requests
import argparse

# def request(url, text, ref_wav_path, ref_text, save_file):
#     data = {
#         'text': text,
#         'text_language': 'all_zh',
#         # 'ref_wav_path': ref_wav_path,
#         # 'ref_text': ref_text,
#     }
#     response = requests.post(url, json=data)
#     with open(save_file, 'wb') as f:
#         f.write(response.content)

# parser = argparse.ArgumentParser()
# parser.add_argument('--url', type=str, required=True)
# parser.add_argument('--text', type=str, required=True)
# parser.add_argument('--ref_wav_path', type=str, required=False, default='')
# parser.add_argument('--ref_text', type=str, required=False, default='')
# parser.add_argument('--save_file', type=str, required=True)
# parser.set_defaults(func=request)
# args = parser.parse_args()
# args.func(args.url, args.text, args.ref_wav_path, args.ref_text, args.save_file)

SOVITS_SERVER_URL = 'http://127.0.0.1:8001/'
STORY_SERVER_URL = 'http://127.0.0.1:8070/'

import sys
import argparse

def infer_server_sovits(args):
    voice_id = args.voice_id
    text = args.text

    if text.startswith('@'):
        with open(text[1:], 'r', encoding='utf8') as f:
            text = f.read()

    data = {
        "text": text,
        "voice_id": voice_id,
    }
    response = requests.post(SOVITS_SERVER_URL+'infer/', json=data)
    print(response.content)

def infer_server_story(args):
    voice_id = args.voice_id
    text = args.text

    if text.startswith('@'):
        with open(text[1:], 'r', encoding='utf8') as f:
            text = f.read()
    data = {
        "text": text,
        "voiceId": voice_id,
    }
    response = requests.post(STORY_SERVER_URL+'ai/test/sovits/infer/', params=data)
    print(response.content)

args = argparse.Namespace(voice_id=sys.argv[1], text=sys.argv[2])
infer_server_story(args)


