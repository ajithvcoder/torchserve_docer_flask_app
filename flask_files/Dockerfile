FROM python:3.9-slim

USER root
RUN apt update -y
WORKDIR /app
EXPOSE 8085
COPY . /app
RUN pip3 install -r requirements.txt


CMD ["python3", "app.py"]