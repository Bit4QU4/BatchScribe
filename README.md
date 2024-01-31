# Goal of this project is to intake audio/video and create a transcript including the speaker and what they said into a text document
# Maybe add the ability to make better captions? might be a diff project though

To enable GPU:
https://github.com/openai/whisper/discussions/47
These steps are tested on Windows 11 with CUDA 11.7 + torch 1.12.1

Download cuda_11.7.0_windows_network.exe network installer from CUDA Toolkit 11.7 Downloads | NVIDIA Developer
Select only Runtime at install.
install pytorch with CUDA: pip3 install torch==2.0.1+cu117 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
Check CUDA availability with python -c 'import torch; print(\"CUDA enabled:\", torch.cuda.is_available());'. if everything OK, the script will output CUDA enabled: True