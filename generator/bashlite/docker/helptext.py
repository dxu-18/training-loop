"""`docker COMMAND --help`: usage and the common options, in our own words."""

HELP = {
    "run": """
Usage:  docker run [OPTIONS] IMAGE [COMMAND] [ARG...]

Create and run a new container from an image

Aliases:
  docker container run, docker run

Options:
      --add-host list      Add a custom host-to-IP mapping (host:ip)
      --cpus decimal       Number of CPUs
  -d, --detach             Run container in background and print container ID
      --entrypoint string  Overwrite the default ENTRYPOINT of the image
  -e, --env list           Set environment variables
      --env-file list      Read in a file of environment variables
      --gpus gpu-request   GPU devices to add to the container ('all' to pass all GPUs)
  -h, --hostname string    Container host name
      --init               Run an init inside the container that forwards signals and reaps processes
  -i, --interactive        Keep STDIN open even if not attached
      --ipc string         IPC mode to use
  -l, --label list         Set meta data on a container
  -m, --memory bytes       Memory limit
      --mount mount        Attach a filesystem mount to the container
      --name string        Assign a name to the container
      --network network    Connect a container to a network
  -p, --publish list       Publish a container's port(s) to the host
  -P, --publish-all        Publish all exposed ports to random ports
      --platform string    Set platform if server is multi-platform capable
      --pull string        Pull image before running ("always", "missing", "never") (default "missing")
      --restart string     Restart policy to apply when a container exits (default "no")
      --rm                 Automatically remove the container and its associated anonymous volumes when it exits
      --shm-size bytes     Size of /dev/shm
      --tmpfs list         Mount a tmpfs directory
  -t, --tty                Allocate a pseudo-TTY
  -u, --user string        Username or UID (format: <name|uid>[:<group|gid>])
  -v, --volume list        Bind mount a volume
  -w, --workdir string     Working directory inside the container
""",
    "ps": """
Usage:  docker ps [OPTIONS]

List containers

Options:
  -a, --all             Show all containers (default shows just running)
  -f, --filter filter   Filter output based on conditions provided
      --format string   Format output using a custom template
  -n, --last int        Show n last created containers (includes all states) (default -1)
  -l, --latest          Show the latest created container (includes all states)
      --no-trunc        Don't truncate output
  -q, --quiet           Only display container IDs
  -s, --size            Display total file sizes
""",
    "images": """
Usage:  docker images [OPTIONS] [REPOSITORY[:TAG]]

List images

Options:
  -a, --all             Show all images (default hides intermediate images)
      --digests         Show digests
  -f, --filter filter   Filter output based on conditions provided
      --format string   Format output using a custom template
      --no-trunc        Don't truncate output
  -q, --quiet           Only show image IDs
""",
    "build": """
Usage:  docker buildx build [OPTIONS] PATH | URL | -

Start a build

Aliases:
  docker build, docker builder build, docker image build, docker buildx b

Options:
      --build-arg stringArray   Set build-time variables
  -f, --file string             Name of the Dockerfile (default: "PATH/Dockerfile")
      --label stringArray       Set metadata for an image
      --no-cache                Do not use cache when building the image
      --platform stringArray    Set target platform for build
      --progress string         Set type of progress output ("auto", "plain", "tty")
      --pull                    Always attempt to pull all referenced images
      --push                    Shorthand for "--output=type=registry"
  -q, --quiet                   Suppress the build output and print image ID on success
      --secret stringArray      Secret to expose to the build (format: "id=mysecret[,src=/local/secret]")
  -t, --tag stringArray         Name and optionally a tag (format: "name:tag")
      --target string           Set the target build stage to build
""",
    "logs": """
Usage:  docker logs [OPTIONS] CONTAINER

Fetch the logs of a container

Options:
  -f, --follow         Follow log output
      --since string   Show logs since timestamp or relative (e.g. 42m for 42 minutes)
  -n, --tail string    Number of lines to show from the end of the logs (default "all")
  -t, --timestamps     Show timestamps
""",
    "exec": """
Usage:  docker exec [OPTIONS] CONTAINER COMMAND [ARG...]

Execute a command in a running container

Options:
  -d, --detach        Detached mode: run command in the background
  -e, --env list      Set environment variables
  -i, --interactive   Keep STDIN open even if not attached
  -t, --tty           Allocate a pseudo-TTY
  -u, --user string   Username or UID (format: "<name|uid>[:<group|gid>]")
  -w, --workdir string  Working directory inside the container
""",
    "inspect": """
Usage:  docker inspect [OPTIONS] NAME|ID [NAME|ID...]

Return low-level information on Docker objects

Options:
  -f, --format string   Format output using a custom template
  -s, --size            Display total file sizes if the type is container
      --type string     Only inspect objects of the given type
""",
}
