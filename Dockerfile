FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.pytorch2

USER root

RUN apt-get update && apt-get install -y curl

USER 1000

RUN pip install pycocotools==2.0.7 

