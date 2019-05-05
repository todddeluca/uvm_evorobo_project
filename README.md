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

Hyperparameters, defined in `make_hyperparameters()` in `run.py` or `scaffolding.py` define and record the parameters of an experiment,
like the number of legs of a robot, type of obstacle (stairs, ladder, lattice), population size, or number of generations to evolve.

### Earlier Experiments

Evolve parameters and save them:

    python run.py train

Play (i.e. run a visible simulation using) the default saved parameters (from `robots.pkl`):

    python run.py play

Evolve parameters starting from an existing set of parameters (or population):

    time python run.py train --restore experiments/exp_20190421_230912_robot.pkl

Play parameters saved in a file:

    time python run.py play --restore experiments/exp_20190421_230912_robot.pkl
    
### Evolved Spatial Scaffolding Experiments

Create and evolve a new population, saving it at checkpoints and the end:

    python scaffolding.py train

Play (i.e. run a visible simulation using) a genome from the default saved population (from `population.pkl`):

    python scaffolding.py play

Evolve population starting from a checkpoint of a saved population:

    time python scaffolding.py train --restore experiments/exp_20190505_024155_population.pkl

Play a genome from the population saved in a file:

    time python scaffolding.py play --restore experiments/exp_20190505_024155_population.pkl
