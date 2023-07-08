FROM mcr.microsoft.com/devcontainers/python:3.9
WORKDIR /workspaces
RUN \
    # devcontainer dependencies
    apt-get update && \
    apt-get install -y git graphviz && \
    # MICrONS dependencies
    pip3 install git+https://github.com/AllenInstitute/em_coregistration.git@phase3 && \
    pip3 install seaborn ipykernel && \
    pip3 install git+https://github.com/yambottle/microns_phase3_nda.git

ENV DJ_HOST db.datajoint.com
ENV DJ_USER microns
ENV DJ_PASS microns2023
ENV DATABASE_PREFIX microns_