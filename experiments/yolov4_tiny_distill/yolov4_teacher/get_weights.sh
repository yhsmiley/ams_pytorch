# Checks if gdown is installed
if ! type "gdown" > /dev/null; then
    read -p "Install gdown (google drive downloader) with pip3 (y/n)? " yn
    case $yn in
        [Yy]* ) pip3 install gdown;;
        * ) echo "Not installing gdown. Exiting.."; exit;;
    esac
fi

gdown -O yolov4.pth.tar https://drive.google.com/uc?id=1DcXXETpLKZHQJgXjpRGtrklFS97H8TY5

gdown -O yolov4-tiny.pth.tar https://drive.google.com/uc?id=1rrfYRpAwJe5NwKXiflDnuiyuRl3MZ8CQ