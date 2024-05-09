# from audio_separator.separator import Separator
#
# class AudioProcessingPipeline:
#     def __init__(self):
#         self.separator = Separator()
#         self.models = {
#             'Kim_Vocals_2': ('Kim_Vocal_2.onnx', {'mdxc_batch_size': 4, 'mdxc_segment_size': 256, 'mdxc_overlap': 8, 'mdxc_pitch_shift': 2}),
#             'Karaoke_Ensemble': ('UVR_MDX_NET_Karaoke.onnx', {'vr_batch_size': 16, 'vr_window_size': 320, 'vr_aggression': 2}),
#             'Reverb467': ('Reverb467.onnx', {'mdx_batch_size': 1, 'mdx_segment_size': 256, 'mdx_overlap': 0.25}),
#             'De-Echo': ('VR_UVR_De_Echo_Aggressive.onnx', {'vr_enable_post_process': True, 'vr_post_process_threshold': 0.1}),
#             'Denoise': ('VR_UVR_Denoise.onnx', {'vr_high_end_process': True}),
#             'Final_Stage': ('VR_Arch_Single_Model_v5_6_HP_Karaoke_UVR.onnx', {'vr_enable_tta': True})
#         }
#
#     def load_model(self, model_name):
#         model_file, params = self.models[model_name]
#         self.separator.load_model(model_filename=model_file)
#         # Apply each parameter individually
#         for param, value in params.items():
#             setattr(self.separator, param, value)  # Assuming the Separator class can dynamically accept params this way
#
#     def process_audio(self, audio_file):
#         # Process through each stage
#         current_input = audio_file
#         for model in self.models:
#             self.load_model(model)
#             current_input = self.separator.separate(current_input)[0]  # Assuming separate method returns a list of output file names
#         return current_input
#
#     def run(self, audio_file):
#         final_output = self.process_audio(audio_file)
#         print(f"Final output stored at: {final_output}")
#
# # Usage
# pipeline = AudioProcessingPipeline()
# pipeline.run('path_to_audio_file.wav')
import time
from audio_separator.separator import Separator

def process_audio_pipeline(audio_file):
    # Process with the first model
    separator = Separator()
    separator.load_model(model_filename='Kim_Vocal_2.onnx')
    _, vocals = separator.separate(audio_file)
    print('+++++++++++++++++++++++++++')
    print(vocals)

    # Load the next model and process the output from the first step
    separator = Separator()
    separator.load_model(model_filename='UVR_MDXNET_KARA_2.onnx')
    chorus, _ = separator.separate(vocals)
    print('after UVR_MDXNET_KARA_2')
    time.sleep(10)

    # Continue the pipeline by loading and processing with each subsequent model
    separator = Separator()
    separator.load_model(model_filename='UVR-DeEcho-DeReverb.pth')
    reverb, _ = separator.separate(chorus)
    print('after DeReverb')
    time.sleep(10)

    separator = Separator()
    separator.load_model(model_filename='UVR-De-Echo-Aggressive.pth')
    de_echo, _ = separator.separate(reverb)
    print('after -De-Echo-Aggressive')
    time.sleep(10)

    separator = Separator()
    separator.load_model(model_filename='UVR-DeNoise.pth')
    _, denoise = separator.separate(de_echo)
    print('after DeNoise')
    time.sleep(10)

    separator = Separator()
    separator.load_model(model_filename='6_HP-Karaoke-UVR.pth')
    final_instrumentals, final_vocals = separator.separate(denoise)
    print('after Karaoke-UVR')
    time.sleep(10)

    return final_vocals, final_instrumentals

# Specify the initial audio file
audio_file = './example.wav'

final_vocals, final_instrumentals = process_audio_pipeline(audio_file)
print(f'Vocals saved at {final_vocals}')
print(f'Instrumentals saved at {final_instrumentals}')