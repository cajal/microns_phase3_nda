FROM python:3.8-buster

# Base packages

RUN apt-get update&& \
    apt-get -y install graphviz build-essential python-dev fish
    
RUN pip install --upgrade pip

# Install jupyter notebook
RUN pip3 install jupyter jupyterlab

# Install viz dependencies
RUN pip3 install matplotlib ipyvolume

# Lock to datajoint 0.12.9
RUN pip3 install datajoint==0.12.9

# Install the outside packages
WORKDIR /src
COPY phase3/ microns_phase3_nda/phase3/
COPY setup.py microns_phase3_nda/.
COPY README.md microns_phase3_nda/.
RUN pip3 install -e microns_phase3_nda/ --use-feature=in-tree-build
RUN pip3 install git+https://github.com/AllenInstitute/em_coregistration.git@phase3

# Set up work environment
COPY notebooks/ /notebooks/tutorials
WORKDIR /notebooks
RUN mkdir workspace/

# Start jupyter notebook
RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter.sh"]