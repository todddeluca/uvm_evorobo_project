# Assignment 2


## Installation


Create and activate a virtual environment:


    export VENVDIR=~/deploy/2019/venvs/evorobo_hw01
    mkdir $VENVDIR
    python -m venv $VENVDIR
    source $VENVDIR/bin/activate
    
Install project requirements:

    pip install -r requirements.txt # install numpy, etc.


Install pyrosim according to the pyrosim README. See https://github.com/ccappelle/pyrosim. 

    cd ~/deploy/2019/
    git clone https://github.com/ccappelle/pyrosim
    cd pyrosim
    ./build.sh
    pip install -e .


## Docker

- https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc
- https://sourabhbajaj.com/blog/2017/02/07/gui-applications-docker-mac/
- https://dev.to/darksmile92/run-gui-app-in-linux-docker-container-on-windows-host-4kde
- http://fabiorehm.com/blog/2014/09/11/running-gui-apps-with-docker/
- libGL errors in docker: https://github.com/jessfraz/dockerfiles/issues/253
- http://gernotklingler.com/blog/howto-get-hardware-accelerated-opengl-support-docker/
- openGL using CPU, not GPU specific drivers: https://github.com/jamesbrink/docker-opengl

Install Docker

Install XQuartz, an X Windowing system for Mac OS X

    brew cask install xquartz
    
Configure Xquartz by enabling 'Preferences > Security > Allow connections from network clients' in XQuartz.

    open -a Xquartz

Reboot so Xquartz will work.

Build docker image for pyrosim

    docker build -t pyrosim:latest .



Setup connection between container and host so container can access the X11 display

Install socat

    brew install socat
    
Run socat

    socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"

Or:
    open -a Xquartz
    xhost + $IP

Run your code in a docker container

    export IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
    
    # 
    # --privileged
    docker run -it --rm -e DISPLAY=$IP:0 -v `pwd`:/app -w /app --name mycontainer pyrosim:latest python sensors.py
    
    # socat version
    docker run -it --rm -e DISPLAY=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}'):0 -v `pwd`:/app -w /app --name mycontainer pyrosim:latest python sensors.py

## Usage






