# Overwrites default library's Swin with my own
echo "Overwriting the library swin.py"
cp -f libraries_replace/swin.py /opt/conda/lib/python3.10/site-packages/mmdet/models/backbones/swin.py
