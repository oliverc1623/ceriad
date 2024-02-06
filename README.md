# cse290k
LLMs + VQAs + autonomous driving + reinforcement learning

Installs:
```pip3 install highway-env```

# cse290k

Installs:
```pip3 install highway-env```

## CARLA Instruction

1. Follow the Nautilus setup tutorial to install the kubernetes command line and get access to the AIEA lab namespace

https://docs.nationalresearchplatform.org/userdocs/start/quickstart/

- If your not in the AIEA lab namespace please email Coen Adler <ctadler@ucsc.edu> or Shreedhar Jangam <sjangam@ucsc.edu>

2. Once you have installed kubernetes command and can view the AIEA lab's namespace pods create the following:
- A persistent volume claim (PVC) with seaweed-nvme for your main storage. 64Gb should suffice if you're just trying to run CARLA without machine learning. Otherwise, 128Gb or greater should suffice. 
- A persistent volume claim for your deployment's cache. I got by with 1Gb, for reference. 

3. Edit `nautilus-files/xgl.yaml`
- Change the name in line 4
- 

4. 


10. Running CARLA 

`./CarlaUE4.sh -carla-rpc-port=4000`

add the flag `-prefernvidia` if carla is not running on the GPU.

`python generate_traffic.py -p 4000 --tm-port 4050`
