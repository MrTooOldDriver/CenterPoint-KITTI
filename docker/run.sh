docker run -it --rm --ipc=host \
    -v /home/ivan:/root/hantao \
    -p 7947:22 \
    -v /mnt/data:/mnt/data \
    --gpus all\
    --name centerpoint-hantao \
    torch:centerpoint-hantao