# Realtime_object_detection

'''bash
conda update -n base -c defaults conda

conda create -n realtime python=3.8 -y

conda activate visa

pip install -r requirements.txt
'''

yolo detect predict model=yolov8l.pt source=0 show=true

source = 0 means camera

python detect.py