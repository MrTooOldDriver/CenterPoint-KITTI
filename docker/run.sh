nvidia-docker run -it --rm --ipc=host \
    -v /mnt/sda1/hantao:/root/hantao \
    -p 7894:22 \
    -v /mnt/12T:/root/data \
    -v /mnt/12T:/mnt/12T \
    --gpus all\
    --cpuset-cpus "27-31" \
    --name centerpoint-hantao \
    torch:centerpoint-hantao