FROM pytorch/torchserve:latest

COPY ./utils utils
COPY ./models models
COPY handler.py handler.py
COPY requirements.txt requirements.txt
USER root

RUN pip3 install -r requirements.txt

EXPOSE 8080
EXPOSE 8081
RUN torchserve --stop
RUN torch-model-archiver -f --model-name spaceship --version 1.0 --serialized-file models/spaceship.onnx --export-path model-store --handler handler.py --extra-files utils/encoder_traindata.pickle -f

CMD ["torchserve", "--start", "--ncs", "--model-store", "model-store", "--models", "spaceship=spaceship.mar"]
