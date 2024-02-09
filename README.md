# cse290k
LLMs + VQAs + autonomous driving + reinforcement learning

Installs:
```pip3 install highway-env```

# cse290k

Installs:
```pip3 install highway-env```

## Nautilus GUI Desktop/CARLA Instruction

### 1. Follow the Nautilus setup tutorial to install the kubernetes command line and get access to the AIEA lab namespace

https://docs.nationalresearchplatform.org/userdocs/start/quickstart/

- If your not in the AIEA lab namespace please email Coen Adler <ctadler@ucsc.edu> or Shreedhar Jangam <sjangam@ucsc.edu>
- Also see this guide for reference before starting 
https://docs.nationalresearchplatform.org/userdocs/running/gui-desktop/

### 2. Once you have installed kubernetes command and can view the AIEA lab's namespace pvc create the following:
- A persistent volume claim (PVC) with seaweed-nvme for your main storage. 64Gb should suffice if you're just trying to run CARLA without machine learning. Note that is is important that you choose seaweed-nvme for your storage because it has a better high read/write performance compared to ceph-rock. 
- A persistent volume claim for your deployment's cache. I got by with 1Gb, for reference. 

### 3. Nautilus' GUI desktop runs in your web browser to port-forward the GUI screen. So you will need to create two secrets for the yaml file below.
- `kubectl create secret generic [my-pass] --from-literal=[my-pass]=[YOUR_PASSWORD]`
- `kubectl create secret generic [turn-shared-secret] --from-literal=[turn-shared-secret]=[MY_TURN_SHARED_SECRET]`
Give your secrets a unique name so there are no conflicts. Consider using your ucsc username.

### 4. Edit `nautilus-files/xgl.yaml`
- Change the name in line 4
- Change line 39, 40 to your secret name and key
- Change line 82 to a username of your choice
- Change line 88, 89 to your secret name and key
- Change line 130 to your cache volume
- Change line 134 to your storage volume

**Before proceeding please ask the nautilus-channel on discord for permission to run a Deployment**

### 5. Run the deployment
`kubectl create -f [your-xgl-filename].yaml`
This deployment should automatically create a pod

### 6. Check if pod is running
- check pod status `kubectl get pods`
If you see your pod is running, then you've completed the hard part!

### 7. Port-forward the pod
Run
-`kubectl port-forward pod/[pod-name] :8080`
You should see an output that looks like this: `Forwarding from 127.0.0.1:[some number] -> 8080`
Copy and paste `127.0.0.1:[some number]` into your web browser 

### 8. Localhost page 
- The page should prompt you to enter the username and password from line 82 and 40 from the deployment's yaml file. 

### 9. Run any program that requires graphics! 

---

### CARLA Installation Instructions

#### 1. Install Miniconda
- https://docs.anaconda.com/free/miniconda/#quick-command-line-install
- Then create a new python environment with Python3.8
    - For example, `conda create -n carla python=3.8`
- Activate your newly made environment

#### 2. Install CARLA
- Follow the install instructions from CARLA's official documentation.
- I would follow the "B. Package Installation" method
https://carla.readthedocs.io/en/latest/start_quickstart/#b-package-installation
- And for the client I followed method C, "Downloadable Python package"

#### 3. Python packages
- Once CARLA is installed or everything is unzipped install the recommended Python packages by running
```
cd PythonAPI\examples
python3 -m pip install -r requirements.txt 
```
It might show an error saying scipy or scikitlearn is not installed. If so, run `pip install scipy` or `pip install scikit-learn`

#### 4. Running CARLA 

`./CarlaUE4.sh -carla-rpc-port=4000` # port num is up to you

add the flag `-prefernvidia` if carla is not running on the GPU.

`python generate_traffic.py -p 4000 --tm-port 4050`
