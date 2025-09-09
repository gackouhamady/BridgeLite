# Build + run the container locally
$ErrorActionPreference = "Stop"
$tag = "bridgelite:local"

docker build -t $tag .
docker run --rm -p 8000:8000 --name bridgelite $tag
