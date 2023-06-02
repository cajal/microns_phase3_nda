FROM python:3.9-slim
WORKDIR /workspaces
RUN \
    # devcontainer dependencies
    apt-get update && \
    apt-get install -y git graphviz && \
    # MICrONS dependencies
    pip3 install git+https://github.com/AllenInstitute/em_coregistration.git@phase3 && \
    pip3 install seaborn ipykernel && \
    git clone https://github.com/yambottle/microns_phase3_nda.git && \
    pip install -e microns_phase3_nda

ENV DJ_HOST tutorial-db.datajoint.io
ENV DJ_USER microns
ENV DJ_PASS microns2021
ENV DATABASE_PREFIX microns_