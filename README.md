# UVM CS206A Evolutionary Robotics Final Project


## Installation


Create and activate a virtual environment:

    export VENVDIR=~/deploy/2019/venvs/evorobo_hw01
    mkdir $VENVDIR
    python -m venv $VENVDIR
    source $VENVDIR/bin/activate

Install project requirements:

    pip install -r requirements.txt # install numpy, etc.

Install pyrosim according to the pyrosim README. See https://github.com/ccappelle/pyrosim:

    cd ~/deploy/2019/
    git clone https://github.com/ccappelle/pyrosim
    cd pyrosim
    ./build.sh
    pip install -e .

Install estool, a suite of evolutionary strategies from hardmaru (David Ha)

    cd ~/deploy/2019
    git clone https://github.com/hardmaru/estool
    cd estool/
    pip install -e .

## Usage

python geneticalgorithm.py







