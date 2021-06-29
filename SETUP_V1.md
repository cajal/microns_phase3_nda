# microns-phase3-nda

This guide will walk you through setting up the database container and the access methods for the data.

# Prerequisites

- ~130 GB of free disk space
- [Docker](https://docs.docker.com/desktop/)
- [docker-compose](https://docs.docker.com/compose/)

# Setup

This section comes in two parts, the first is the database containing the `microns-phase3-nda` schema and the second is for the container to access that data with DataJoint in a Jupyter notebook server that will come with tutorials or with the mysql-client.

If you know what you're doing and wish to handle importing the SQL file into an existing database yourself you can skip this next `Database` section and go straight to the `Access` section.

# Database

The docker image must first be downloaded from this link (in a tar archive format): [mysql-nda-database](LINK_GOES_HERE).
Save this to an accessible location.

In the location where you've stored the downloaded image archive you then will load the image to your local filesystem:

```bash
docker load < database_container_image_v1.tar
```

OR

```bash
docker load --input database_container_image_v1.tar
```

To start the database you can either `Docker` or `docker-compose`:

## Docker

```bash
docker run --network="host" -p 3306:3306 -it database_container_image_v1 /bin/bash -c "find /var/lib/mysql -type f -exec touch {} \; && service mysql start && sleep 1000000000" &
```

# Access

The data can be accessed in two ways, either with the mysql-client or through DataJoint in a Jupyter notebook service.

The default user and password for the image are:

`username:` root  
`password:` microns123

## Jupyter Notebook (DataJoint)

You can clone the repository and build it yourself with `Docker` and `docker-compose`.
Clone the repository at https://github.com/cajal/microns_phase3_nda.

The pre-built image can also be downloaded at this link: [microns-phase3-nda-notebook](LINK_GOES_HERE)

Using the docker-compose you can start the service with:

```bash
docker-compose up -d notebook
```

which can then be accessed at http://localhost:8888/tree (or at a custom port set with env variable NOTEBOOK_PORT or set in an .env file in the same location as the docker-compose.yml).

http://localhost:8888 will send to Jupyter Lab, but it's not guaranteed that all plots/graphics will work out of the box without enabling the relevant jupyter lab extensions.

The database host will default to http://localhost:3306, in order to point to the mysql database you should either set an env variable of the name DJ_HOST (or in the .env file) or use the below Python snippet to set the host before loading the schema.

```env
DJ_HOST="host-ip-goes-here"
```

OR

```python
import datajoint as dj

dj.config['database.host'] = "host-ip-goes-here"

import nda
```

## mysql-client

From the local machine you can access it this way

```bash
mysql -h 127.0.1.1 -u root -p
```

which will then prompt for the password (the default from above is `microns123`) and will open an interactive mysql terminal.