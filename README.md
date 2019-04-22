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

Evolve parameters and save them:

    python run.py train

Play (i.e. run a visible simulation using) the default saved parameters (from `robots.pkl`):

    python run.py play

Evolve parameters starting from an existing set of parameters (or population):

    time python run.py train --restore experiments/exp_20190421_230912_robot.pkl

Play parameters saved in a file:

    time python run.py play --restore experiments/exp_20190421_230912_robot.pkl

