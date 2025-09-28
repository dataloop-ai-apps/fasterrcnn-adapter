FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.pytorch2

USER root

RUN apt-get update && apt-get install -y curl

USER 1000

# Install PyTorch from custom index
RUN pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

RUN pip install pycocotools==2.0.7 \
    numpy==1.23.5 \
    opencv-python==4.8.1.78 \
    pillow==9.2.0

