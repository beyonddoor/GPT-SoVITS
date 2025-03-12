
import requests
import argparse

def request(url, text, ref_wav_path, ref_text, save_file):
    data = {
        'text': text,
        'text_language': 'all_zh',
        # 'ref_wav_path': ref_wav_path,
        # 'ref_text': ref_text,
    }
    response = requests.post(url, json=data)
    with open(save_file, 'wb') as f:
        f.write(response.content)

parser = argparse.ArgumentParser()
parser.add_argument('--url', type=str, required=True)
parser.add_argument('--text', type=str, required=True)
parser.add_argument('--ref_wav_path', type=str, required=False, default='')
parser.add_argument('--ref_text', type=str, required=False, default='')
parser.add_argument('--save_file', type=str, required=True)
parser.set_defaults(func=request)
args = parser.parse_args()
args.func(args.url, args.text, args.ref_wav_path, args.ref_text, args.save_file)
