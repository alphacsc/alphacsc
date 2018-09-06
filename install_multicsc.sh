
pip install -e .
cd alphacsc/other/kmc2
pip install -e .
cd ../sporco
pip install -e .
cd ../soft-dtw
make cython
pip install -e .
cd ../../..