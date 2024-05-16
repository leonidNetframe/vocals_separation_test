# import torch
# import gc
# from audio_separator.separator import Separator
#
# def clear_gpu():
#     """Clears GPU memory by emptying the cache and collecting garbage."""
#     torch.cuda.empty_cache()
#     gc.collect()
#
# def process_audio_pipeline(audio_file):
#     # Process with the first model
#     # separator = Separator()
#     # separator.load_model(model_filename='Kim_Vocal_2.onnx')
#     # _, vocals = separator.separate(audio_file)
#     # print('+++++++++++++++++++++++++++')
#     # print(vocals)
#     #
#     # # Load the next model and process the output from the first step
#     # separator = Separator()
#     # separator.load_model(model_filename='UVR_MDXNET_KARA_2.onnx')
#     # chorus, _ = separator.separate(vocals)
#     #
#     # del separator
#     # clear_gpu()
#
#     # # Continue the pipeline by loading and processing with each subsequent model
#     # separator = Separator()
#     # separator.load_model(model_filename='UVR-DeEcho-DeReverb.pth')
#     # reverb, _ = separator.separate(chorus)
#
#     # del separator
#     # clear_gpu()
#
#     separator = Separator()
#     separator.load_model(model_filename='UVR-De-Echo-Aggressive.pth')
#     de_echo, _ = separator.separate(audio_file)
#
#     del separator
#     clear_gpu()
#
#     separator = Separator()
#     separator.load_model(model_filename='UVR-DeNoise.pth')
#     _, denoise = separator.separate(de_echo)
#
#     del separator
#     clear_gpu()
#
#     separator = Separator()
#     separator.load_model(model_filename='6_HP-Karaoke-UVR.pth')
#     final_instrumentals, final_vocals = separator.separate(denoise)
#
#     return final_vocals, final_instrumentals


import torch
import gc
from audio_separator.separator import Separator


def clear_gpu():
    """Clears GPU memory by emptying the cache and collecting garbage."""
    torch.cuda.empty_cache()
    gc.collect()


def process_audio_pipeline(audio_file):
    separator = Separator()

    # Function to load model and process audio
    def process_with_model(separator, model_filename, input_audio):
        separator.load_model(model_filename=model_filename)
        output1, output2 = separator.separate(input_audio)
        return output1, output2

    # Process with the first main model
    vocals, _ = process_with_model(separator, 'Kim_Vocal_2.onnx', audio_file)
    print('+++++++++++++++++++++++++++')
    print(vocals)
    clear_gpu()

    chorus, _ = process_with_model(separator, 'UVR_MDXNET_KARA_2.onnx', vocals)
    clear_gpu()

    reverb, _ = process_with_model(separator, 'UVR-DeNoise-Lite.pth', chorus)
    clear_gpu()

    de_echo, _ = process_with_model(separator, 'UVR-De-Echo-Aggressive.pth', reverb)
    clear_gpu()

    _, denoise = process_with_model(separator, 'UVR-DeNoise.pth', de_echo)
    clear_gpu()

    final_instrumentals, final_vocals = process_with_model(separator, '6_HP-Karaoke-UVR.pth', denoise)

    return final_vocals, final_instrumentals

separator = Separator()
# Specify the initial audio file
audio_file = 'example.wav'
# Run the full audio processing pipeline
final_vocals, final_instrumentals = process_audio_pipeline(audio_file)
print(f'Vocals saved at {final_vocals}')
print(f'Instrumentals saved at {final_instrumentals}')