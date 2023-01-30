FROM python:3.9-slim

LABEL maintainer="Jibinraj Antony <jibinraj.antony@dfki.de>"

ENV DEBIAN_FRONTEND = noninteractive

RUN apt-get update && yes | apt-get upgrade

RUN  apt-get install ffmpeg libsm6 libxext6 curl  -y

RUN mkdir -p /odapi

RUN apt-get install -y python3-pip

WORKDIR /odapi

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["detection_app.py"]