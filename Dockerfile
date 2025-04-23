FROM ubuntu:latest
LABEL authors="trant"

ENTRYPOINT ["top", "-b"]