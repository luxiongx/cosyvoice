# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import signal
import traceback
import argparse
import requests
import urllib3
from datetime import datetime, timedelta
from fastapi import FastAPI, APIRouter,Body,Request
import  uvicorn
import fastapi_cdn_host
from typing import Optional
from pydantic import BaseModel
import numpy as np
import torch
import torchaudio
import random
import librosa
import soundfile as sf  
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

APP = FastAPI()

max_val = 0.8
stream_mode_list = [('否', False)]

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech





class bm_tts(BaseModel):
    voice: Optional[str] =""
    text: Optional[str] =""
    address: Optional[str] ="./"
    

@APP.post("/tts",summary='图片to视频')
def generate_audio(inputData:bm_tts=Body(...,example={'voice': "test",
                                                        'text': "测试文本",
                                                        'address':""
                                                        })):
    tts_text=inputData.text
    voice=inputData.voice
    address=inputData.address
    speed=1
    content=""
    wavname=address+"/voice/"+voice+"/"+voice+".wav"
    txtname=address+"/voice/"+voice+"/"+voice+".txt"
    
    now = datetime.now()
    temp_name = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    
    
    filename=address+"/music/"+temp_name+".wav"
    if os.path.exists(txtname):        
        with open(txtname, 'r') as file:
            content = file.read()
                      
    
    


    if wavname is None:
        print('prompt音频为空，您是否忘记输入prompt音频？')
        yield (cosyvoice.sample_rate, default_data)
    if torchaudio.info(wavname).sample_rate < prompt_sr:
        print('prompt音频采样率{}低于{}'.format(torchaudio.info(wavname).sample_rate, prompt_sr))
      
        yield (cosyvoice.sample_rate, default_data)
    full_audio = np.array([], dtype=np.float32)
    logging.info('get zero_shot inference request')
    prompt_speech_16k = postprocess(load_wav(wavname, prompt_sr))
    set_all_random_seed(0)
    for i in cosyvoice.inference_zero_shot(tts_text, content, prompt_speech_16k, stream=False, speed=speed):
        #yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
        chunk = i['tts_speech'].numpy().flatten().astype(np.float32)
        full_audio = np.append(full_audio, chunk)
    
    # 如需实时保存每个分块(可选)
    # sf.write(f"chunk_{i}.wav", chunk, cosyvoice.sample_rate)

# 保存完整音频文件
    sf.write(filename, full_audio, cosyvoice.sample_rate)
   
    #with open(filename, 'wb') as f:
    #    f.write(full_audio)  # 注意：response.data可能不是所有Flask版本都支持，直接使用audio_data更常见

    return JSONResponse(status_code=200, content={"message": filename})
    
    
    
ip_address = "0.0.0.0"  # Replace with your desired IP address
port_number = 7777  # Replace with your desired port number

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=7777)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    try:
        uvicorn.run(APP, host=ip_address, port=port_number, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)   
