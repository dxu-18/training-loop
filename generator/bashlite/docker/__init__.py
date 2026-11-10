"""docker-lite: the Docker engine and CLI on bash-lite (images, containers, builds, volumes, networks, registries,
compose). Importing it registers the `docker` program and the programs that live inside images."""

from . import cli, cmd_images, cmd_misc, build, compose, programs, state  # noqa: F401
