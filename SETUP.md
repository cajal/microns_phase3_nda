# microns-phase3-nda

This guide will walk you through setting up the database container and the access methods for the data.

# Prerequisites

- ~115 GB of free disk space (around double that, ~220 GB, to load the image from the tar archive the first time)
- [Docker](https://docs.docker.com/desktop/)
- [docker-compose](https://docs.docker.com/compose/)

# Setup

This section comes in two parts, the first is the database containing the `microns-phase3-nda` schema and the second is for the container to access that data with DataJoint in a Jupyter notebook server that will come with tutorials or with the mysql-client.

If you know what you're doing and wish to handle importing the SQL file into an existing database yourself you can skip this next `Database` section and go straight to the `Access` section.

# Database

The docker image must first be downloaded from this link (in a tar archive format): [mysql-nda-database](LINK_GOES_HERE).
Save this to an accessible location.

In the location where you've stored the downloaded image archive you then will load the image to your local filesystem (replacing image-name with the name of the downloaded file):

```bash
docker load < image-name.tar
```

OR

```bash
docker load --input image-name.tar
```

To start the database you can either `Docker` or `docker-compose`:

The data is this database, started with both Docker or docker-compose, is not persistent and changes will be lost when exiting the container.

## Docker

Again replacing image-name with the proper name.

```bash
docker run --network="host" --detach image-name
```

## docker-compose

This would be the preferred method for starting the service.
```bash
docker-compose up -d database
```

# Access

The data can be accessed in two ways, either with the mysql-client or through DataJoint in a Jupyter notebook service.

The default user and password for the image are:

`username:` root  
`password:` microns123

## Jupyter Notebook (DataJoint)

The pre-built image can be downloaded at this link: [microns-phase3-nda-notebook](LINK_GOES_HERE)

You can also clone the repository and build it yourself with `Docker` and `docker-compose`.
Clone the repository at https://github.com/cajal/microns_phase3_nda.

Using the docker-compose you can start the service with:

```bash
docker-compose up -d notebook
```
which can then be accessed at http://localhost:8888/tree (or at a custom port set with env variable NOTEBOOK_PORT or set in an .env file in the same location as the docker-compose.yml).

http://localhost:8888 will send to Jupyter Lab, but not all plots/graphics will work out of the box without enabling jupyter lab extensions.

The database host will default to http://localhost:3306, but that points inside the container, in order to point to the mysql database you should either set an env variable of the name DJ_HOST (or in the .env file) or use the below Python snippet to set the host before loading the schema.

An external, persistent workspace can be mounted to the internal `workspace` folder by settings the `EXTERNAL_NOTEBOOKS` env variable to a folder of choice.

```python
import datajoint as dj

# This will be 127.0.1.1 if the container is on the same machine as the database, or just the hostname of the machine the database lives on.
dj.config['database.host'] = "database-host-ip-goes-here"

from phase3 import nda, func, utils
```

## mysql-client

From the local machine you can access it this way

```bash
mysql -h 127.0.1.1 -u root -p
```

which will then prompt for the password (the default from above is `microns123`) and will open an interactive mysql terminal.

The external hostname of the machine can be used in place of `127.0.1.1`.