import gc
import html
import io
import os
import queue
import wave
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
import traceback
import ssl
import uuid
import time
import json

import tempfile
import subprocess

import gradio as gr
import librosa
import numpy as np
import pyrootutils
import torch
from loguru import logger
from transformers import AutoTokenizer

# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from fish_speech.i18n import i18n
# from fish_speech.text.chn_text_norm.text import Text as ChnNormedText

from infer_single_nos import infer_tools, AttrDict



import threading
import torch.multiprocessing as mp # 多线程


# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"
# manager = mp.Manager()

# mp.set_start_method('spawn', force=True)

HEADER_MD = f"""# kalle

{i18n("Kalle kxxia")}  

"""

TEXTBOX_PLACEHOLDER = i18n("Put your text here.")
SPACE_IMPORTED = False


def build_html_error_message(error):
    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


@torch.inference_mode()
def inference(
    reference_audio,
    reference_text,
    refined_text,
    enable_reference_audio,
):

    try:   
        print(f'refined_text: {refined_text}')
        print(f'reference_text: {reference_text}')
        audio = model.infer(reference_audio, reference_text, refined_text, enable_reference_audio)
        # audio = audio.detach().cpu().numpy().astype('int16')
        # audio = audio.detach().cpu().numpy()1
        
        result = None, (16000, audio), "no error"
        # return result[0], result[1], result[2]
        return result[1], result[2]
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)  # 记录异常信息和堆栈
        logger.error("错误详细信息:\n" + traceback.format_exc())  # 记录完整的异常堆栈信息
        return None, f"error:{e}"
    
def inference_optim(
    reference_audio,
    reference_text,
    refined_text,
    enable_reference_audio,
):

    try:   
        print(f'refined_text: {refined_text}')
        print(f'reference_text: {reference_text}')
        audio = model_optim.infer(reference_audio, reference_text, refined_text, enable_reference_audio)
        # audio = audio.detach().cpu().numpy().astype('int16')
        # audio = audio.detach().cpu().numpy()1
        
        result = None, (16000, audio), "no error"
        # return result[0], result[1], result[2]
        return result[1], result[2]
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)  # 记录异常信息和堆栈
        logger.error("错误详细信息:\n" + traceback.format_exc())  # 记录完整的异常堆栈信息
        return None, f"error:{e}"
    
    
def check_audio_validity(wav_data):
    # 使用临时文件来保存音频数据
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_wav.write(wav_data)
        temp_wav.close()
        
        # 使用 ffmpeg 检查音频文件的有效性
        try:
            # 调用 ffmpeg 命令来检查音频文件
            ffmpeg_command = [
                'ffmpeg', '-v', 'error', '-i', temp_wav.name, '-f', 'null', '-'
            ]
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"音频文件 {temp_wav.name} 验证成功。")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"音频文件 {temp_wav.name} 无效。FFmpeg 错误: {e.stderr.decode()}")
            return False
        finally:
            # 删除临时文件
            if os.path.exists(temp_wav.name):
                os.remove(temp_wav.name)
                



n_audios = 1

global_audio_list = []
global_error_list = []



def wav_chunk_header(sample_rate=48000, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


# def normalize_text(user_input, use_normalization):
#     if use_normalization:
#         return ChnNormedText(raw_text=user_input).normalize()
#     else:
#         return user_input


def normalize_me(user_input):
    return user_input
# asr_model = None


def build_app():
    global model
    global devive
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % args.theme,
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=5
                )
                refined_text = gr.Textbox(
                    label=i18n("Realtime Transform Text"),
                    placeholder=i18n(
                        "Normalization Result Preview (Currently Only Chinese)"
                    ),
                    lines=5,
                    interactive=False,
                    visible= False
                )

                with gr.Row():
                    if_refine_text = gr.Checkbox(
                        label=i18n("Text Normalization"),
                        value=False,
                        scale=1,
                        visible= False
                    )

                with gr.Row():
                    with gr.Tab(label=i18n("Advanced Config")):

                        # 分组：LLAMA 参数
                        with gr.Row():
                            gr.Markdown("### optim参数")  # 标题在上
                            with gr.Column():
                                top_k = gr.Slider(
                                    label=i18n("Train_Max_step"),
                                    minimum=1,
                                    maximum=400,
                                    value=200,
                                    step=1,
                                    interactive=True
                                )
                                top_p = gr.Slider(
                                    label="Scheduler_Warmup_step",
                                    minimum=1,
                                    maximum=100,
                                    value=10,
                                    step=1,
                                    interactive=True
                                )
                                repetition_penalty = gr.Slider(
                                    label=i18n("Scheduler_Training_steps"),
                                    minimum=1,
                                    maximum=400,
                                    value=150,
                                    step=1,
                                    interactive=True
                                )
                                temperature = gr.Slider(
                                    label="Learning_rate",
                                    minimum=1e-9,
                                    maximum=1,
                                    value=1e-4,
                                    step=1e-8,
                                    interactive=True
                                )

                #         # 分组：Stream 流式生成参数
                #         with gr.Row():
                #             gr.Markdown("### 流式生成参数")  # 标题在上
                #             with gr.Column():
                #                 chunk_size = gr.Slider(
                #                     label="chunk_size",
                #                     minimum=4,
                #                     maximum=8,
                #                     value=4,
                #                     step=1,
                #                     interactive=True
                #                 )
                #                 token_cache_length = gr.Slider(
                #                     label="token_cache_length",
                #                     minimum=2,
                #                     maximum=4,
                #                     value=2,
                #                     step=1,
                #                     interactive=True
                #                 )


                    with gr.Tab(label=i18n("Reference Audio")):
                        gr.Markdown(
                            i18n(
                                "a reference audio, useful for specifying speaker."
                            )
                        )

                        enable_reference_audio = gr.Checkbox(
                            label=i18n("Enable Reference Audio"),
                        )
                        
                        ### 这个是预设的参考音频
                        # Add dropdown for selecting example audio files
                        examples_dir = Path("prompt_dir")
                        if not examples_dir.exists():
                            examples_dir.mkdir()
                        example_audio_files = [
                            f.name for f in examples_dir.glob("*.wav")
                        ] + [f.name for f in examples_dir.glob("*.mp3")]
                        example_audio_dropdown = gr.Dropdown(
                            label=i18n("Select Example Audio"),
                            choices=[""] + example_audio_files,
                            value="",
                        )

                        reference_audio = gr.Audio(
                            label=i18n("Reference Audio"),
                            type="filepath",
                            interactive=True,
                        )
                        reference_text = gr.Textbox(
                            label=i18n("Reference Text"), placeholder=i18n("Enter the text corresponding to the reference audio"), lines=5
                        )


            with gr.Column(scale=3):
                # for _ in range(n_audios):
                with gr.Row():
                    # error = gr.HTML(
                    #     label=i18n("Error Message"),
                    #     visible=True,
                    # )
                    error = gr.Text(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                    global_error_list.append(error)
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )
                    global_audio_list.append(audio)

                # with gr.Row():
                #     stream_audio = gr.Audio(
                #         label=i18n("Streaming Audio"),
                #         streaming=True,
                #         autoplay=True,
                #         interactive=False,
                #         show_download_button=True
                #     )
                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001F3A7 " + i18n("Generate"), variant="primary"
                        )
                        
                        generate_optim = gr.Button(
                            value="\U0001F3A7 " + i18n("Generate optim"), variant="primary"
                        )
                        
                        # generate_stream = gr.Button(
                        #     value="\U0001F3A7 " + i18n("Streaming Generate"),
                        #     variant="primary",
                        # )

        text.input(
            fn=normalize_me, inputs=[text], outputs=[refined_text]
        )

        def select_example_audio(audio_file):
            if audio_file:
                audio_path = examples_dir / audio_file
                
                # 构建同名TXT文件路径（替换扩展名）
                txt_path = examples_dir / audio_file.replace('.wav', '.txt').replace('.mp3', '.txt')
                
                # 读取TXT文件内容（如果存在）
                txt_content = ""
                if txt_path.exists():
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            txt_content = f.read().strip()
                    except Exception as e:
                        print(f"读取TXT文件失败: {e}")
                
                return str(audio_path), True, txt_content
            return None, False, ""

        # Connect the dropdown to update reference audio and text
        example_audio_dropdown.change(
            fn=select_example_audio,
            inputs=[example_audio_dropdown],
            outputs=[reference_audio, enable_reference_audio, reference_text],
        )


        generate.click(
            inference,
            [
                reference_audio,
                reference_text,
                text,
                enable_reference_audio,
                
            ],
            [global_audio_list[0], global_error_list[0]]
        )
        
        
        generate_optim.click(
            inference_optim,
            [
                reference_audio,
                reference_text,
                text,
                enable_reference_audio,
                
            ],
            [global_audio_list[0], global_error_list[0]]
        )
        
        # generate_stream.click(
        #     inference_stream,
        #     [
        #         reference_audio,
        #         text,
        #         enable_reference_audio,
        #         temperature,
        #         top_k,
        #         top_p,
        #         repetition_penalty,
        #         chunk_size,
        #         token_cache_length
        #     ],
        #     [stream_audio, global_audio_list[0], global_error_list[0]]
            # concurrency_limit=10,
        # )

    return app


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.4",
    )
    # parser.add_argument(
    #     "--decoder-checkpoint-path",
    #     type=Path,
    #     default="checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    # )
    # parser.add_argument("--decoder-config-name", type=str, default="firefly_gan_vq")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device2", type=str, default="cuda:1")
    # parser.add_argument("--half", action="store_true")
    # parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    global devive
    global device2
    global model
    global model_optim
    device = '0'
    device2 = '0'
    # config = 'configs/vae_12_5_dim1024-sft.yaml'
    config = 'configs/vae_12_5hz_dim2048_tts-sft.yaml'
    # check_point_path = '../epoch_43_step_105353.pt-melvae_1024dim_12_5hz_tts-2000hsft'
    check_point_path = '../epoch_41_step_181058.pt-vae_12_5hz_dim2048_tts-sft'
    
    model = infer_tools(config, device, check_point_path, False, False)

    model_optim = infer_tools(config, device2, check_point_path, False, True)
    
    logger.info("Launching the web UI...")

    # # 创建 SSLContext
    # context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    # context.load_cert_chain(certfile="/etc/ssl/mycert/mydomain.crt", keyfile="/etc/ssl/mycert/mydomain.key")

    app = build_app()
    app.launch(show_api=True,share=False, server_name= "0.0.0.0", server_port = 7861)
    # app.launch(show_api=True,share=False)
