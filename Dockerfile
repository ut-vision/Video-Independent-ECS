FROM python:3.7-slim

RUN apt-get -y update
# for dlib
RUN apt-get install -y build-essential cmake
# for opencv
RUN apt-get install -y libopencv-dev

RUN pip install --upgrade pip
# pip instlal
RUN pip install dlib \
  && pip install opencv-python

RUN pip install scikit-learn

RUN pip install scikit-spatial

RUN pip install seaborn

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install tensorflow

RUN pip install imutils

