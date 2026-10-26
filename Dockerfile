FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
WORKDIR /challenge
COPY requirements.txt /challenge/
RUN pip install --no-cache-dir torchvision==0.17.2 && pip install --no-cache-dir -r requirements.txt
COPY . /challenge
RUN pip install --no-deps -e media
# Weights and data are mounted at runtime. Building never downloads model weights.
