# CARLA Data Collector for LLaVA Fine-tuning

1. Run `./CarlaUE4.sh`
2. Configure the simulator's settings by changing the map, weather, and NPCs
For example, to spawn NPC 50 vehicles and 30 pedestrians run. Remember to change the TM-port since the Nautilus GUI is using the default port, 8000.
`/{CARLA home directory}/PythonAPI/examples/generate_traffic.py -n 50 -w 30 --tm-port 5000`
3. In this directory, run `llava-data-collector.py`. This file spawns an ego vehicle, along with other sensors, in the CARLA world and turns on autopilot. 
The ego agent will roam around the map, following basic driving and traffic rules. 
This file will also create two things: a json file and image output directory. This json file and image directory will be used to fine-tune llava.

The json file should look something like this:
```{json}
[{
    "id": 241386,
    "file_path": "output/241386.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nI am the driver driving at 85.0 mph and steering at -0.8611 degrees. What are the vehicles in the image, their positions, and are they a possible threat to the ego vehicle based on their driving?"
        },
        {
            "from": "gpt",
            "value": " No direct obstacles detected."
        }
    ]
},{
    "id": 241406,
    "file_path": "output/241406.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nI am the driver driving at 45.8231 mph and steering at -3.4085 degrees. What are the vehicles in the image, their positions, and are they a possible threat to the ego vehicle based on their driving?"
        },
        {
            "from": "gpt",
            "value": " No direct obstacles detected."
        }
    ]
},...]
```