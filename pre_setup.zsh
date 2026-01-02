#' Virtual Environment
/usr/local/bin/python3 .venv venv
source .venv/bin/activate
pip install --upgrade pip

pip install -U pip setuptools wheel pybind11


#' Homeomorphic Encryption Library
pip install Pyfhel==3.5.0

#' Install before reqs (causes issues otherwise)
pip install "oprf==5.0.0" "oblivious>=7,<8"
