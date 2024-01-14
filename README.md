# Torchserve_docer_flask_app Integration

This repo tells how to use two different docker containers where flask runs in one container and torch serve runs in another container and make a connection between them using docker compose. (Deployed in AWS)

### Usage

- docker compose up

- docker compose down

### Visuvalization

- http://localhost:8085


### AWS Install docker

    sudo apt-get update -y

    sudo apt-get upgrade

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker

### Note
- Dont forget to terminate the EC2 instance