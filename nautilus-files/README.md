# Desktop GUI Setup for simulator environments

## Contents

- [Nautilus GUI Desktop](#nautilus-gui-desktop)
- [CARLA Installation](#carla-installation-instructions)
- [Common Problems](#common-errors-that-you-might-face)

## Nautilus GUI Desktop

### 1. Follow the Nautilus setup tutorial to install the kubernetes command line and get access to the AIEA lab namespace

https://docs.nationalresearchplatform.org/userdocs/start/quickstart/

- If your not in the AIEA lab namespace please email Coen Adler <ctadler@ucsc.edu> or Shreedhar Jangam <sjangam@ucsc.edu>
- Also see this guide for reference before starting 
https://docs.nationalresearchplatform.org/userdocs/running/gui-desktop/

### 2. Once you have installed kubernetes command and can view the AIEA lab's namespace pvc create the following:
- A persistent volume claim (PVC) with seaweed-nvme for your main storage. 64Gb should suffice if you're just trying to run CARLA without machine learning. Note that is is important that you choose seaweed-nvme for your storage because it has a better high read/write performance compared to ceph-rock. 
- A persistent volume claim for your deployment's cache. I got by with 1Gb, for reference. 

### 3. Once the PVCs are created you need to ask a turn-shared secret from Nautilus Support
- Once you're done setting up your pvcs head on to the nautilus support channel and ask them for a turn secret. Someone from their team will ping and share the key to you. Keep that key with you, you will require it at a later stage when creating your secrets.
- The turn-shared secret is required if you're setting up a server instead of port forwarding your pod everytime. For more details you can check out this page - https://docs.nationalresearchplatform.org/userdocs/running/gui-desktop/#secret-generation

### 4. Nautilus' GUI desktop runs in your web browser to port-forward the GUI screen. So you will need to create two secrets for the yaml file below.
- `kubectl create secret generic [my-pass] --from-literal=[my-password]=[YOUR_PASSWORD] --from-literal=[turn-secret]=[TURN_SHARED_SECRET_FROM_NAUTILUS]`
- `kubectl create secret generic [turn-shared-secret] --from-literal=[turn-shared]=[MY_TURN_SHARED_SECRET]`
<br />Give your secrets a unique name so there are no conflicts. Consider using your ucsc username.
<br />Update your values appropriately at places where its enclosed in square brackets (Don't use square brackets in your command) 
<br />The first secret generates your authentication to the GUI Desktop and the second secret will be your turn-share server authentication details, if you're planning to set that up. Check the docs on Step 3 to know how to do it

### 5. Edit `nautilus-files/xgl.yaml`
- Change the name in line 4
- Change line 39, 40 to your secret name and key (my-pass and my-password in the command provided)
- Change line 79, 80 to your secret name and key (my-pass and turn-secret in the command provided)
- Change line 82 to a username of your choice (This would be your turn-server username)
- Change line 88, 89 to your secret name and key (turn-shared-secret and turn-shared in the command provided)
- Change line 130 to your cache volume
- Change line 134 to your storage volume

**Before proceeding please ask the nautilus-channel on discord for permission to run a Deployment**

### 6. Run the deployment
`kubectl create -f [your-xgl-filename].yaml`
This deployment should automatically create a pod

### 7. Check if pod is running
- Check pod status `kubectl get pods`
- If you face any issues that occur regarding your pod, check and debug using this command `kubectl describe pod [pod-name]`
<br />If you see your pod is running, then you've completed the hard part!

### 8. Port-forward the pod
Run - `kubectl port-forward pod/[pod-name] :8080`
<br />You should see an output that looks like this: `Forwarding from 127.0.0.1:[some number] -> 8080`
<br />Copy and paste `127.0.0.1:[some number]` into your web browser 

### 9. Localhost page 
- The page should prompt you to enter the username which is "user" and password from line 40 of your secret from the deployment's yaml file. 

### 10. Run any program that requires graphics! 

## CARLA Installation Instructions

#### Disclaimer
When installing Carla use the persistent value path `cd ../../mnt/persistent/` from the terminal, to have your data present whenever you port-forward the pod. If you don't do that, then you may lose your data every time you stop your connection.

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

## Common Errors that you might face
- If you're having vulkan driver issues when running Carla on the Desktop GUI, try restarting or create a new deployment.
- If you're facing "Transport endpoint is not connected" when your running cd command, try recreating your volumes or unmount them and run the pod again.
- If you login and are stuck on a black screen, open the side menu and reduce the video bitrate (you can increase this after the GUI shows properly). 
- Sometimes the GUI will freeze all of a sudden. You can fix this by exiting the port-forward, ssh-ing into the pod, run `htop`, then kill the program `selkies-gstreamer` in white text using the most CPU. Exit the ssh pod and start a new port-forward.
- CARLA, by default, uses port 8000 for any Traffic Manager related function calls. However, the Nautilus GUI is using port 8000. Thus, you will have to specify a new Traffic Manager Port.
- You can request certain GPUs like in any pod in the yaml file.
- For more information regarding this deployment use this link - https://docs.nationalresearchplatform.org/userdocs/running/gui-desktop/