# Overcooked-AI
We suppose to apply traditional offline reinforcement learning technique to multi-agent algorithm.  
In this repository, we implemented behavior cloning(BC), offline MADDPG, MADDPG+REM (MADDPG w/ REM), MADDPG+BCQ (MADDPG w/ BCQ) with pytorch.
Now, BCQ is in ' Working In Progress', and it's not implemented completely.  

We collected 0.5M multi-agent offline RL dataset and experimented with each comparison methods. We collected this data with online MADDPG agents, and it includes exploration trajectories using OU noise.
The experiments are ran on  `Asymmetric Advantages` on the Overcooked environment. 


We are looking forward your contribution!


## How to Run
### Collect Offline Data
```bash
python train_online.py agent=maddpg save_replay_buffer=true
```
While the agents train with 0.5M steps, the trajectory replay buffer will be dumped in your `experiment/{date}/{time}_maddpg_{exp_name}/buffer` folder.  
Please replace the path in `config/data/local.yaml` to the experiment by-product directory.


### Download Dataset
Or, if you want to use our dataset pre-collected, please enjoy this [link](https://gisto365-my.sharepoint.com/:u:/g/personal/inchang_baek_gm_gist_ac_kr/EdHtQXOgtyxKnRR8Im4XzckBB1L3RJqJsyFRwTHz76rkWA?e=aYb80b).  
We provide 0.5M trajectories in `Asymmetric Advantages` layout.  
Please download our dataset in your local computer and replace the path in `config/data/local.yaml`

### Train Offline Models
#### Behavior Cloning
```bash
python train_bc.py agent=bc data=local
```

#### Offline MADDPG (Vanilla)
```bash
python train_offline.py agent=maddpg data=local
```
#### Offline MADDPG (w/ REM)
```bash
python train_offline.py agent=rem_maddpg data=local
```
#### Offline MADDPG (w/ BCQ) (WIP)
```bash
python train_offline.py agent=bcq_maddpg data=local
```

### Result
#### Graph
| Online | Offline (0.5M Data) | Offline (0.25M Data) |
| ------------- | ------------- | --- |
|<img src="https://github.com/bic4907/Overcooked-AI/blob/main/result/online_maddpg.png?raw=true" alt="Online MADDPG" width="230"/>|<img src="https://github.com/bic4907/Overcooked-AI/blob/main/result/offline_maddpg.png?raw=true" alt="Full Offline MADDPG" width="230"/>|<img src="https://github.com/bic4907/Overcooked-AI/blob/main/result/half_offline_maddpg.png?raw=true" alt="Half Offline MADDPG" width="230"/>|

#### Video
| Online | BC | Offline /w REM |
| ------------- | ------------- | --- |
|<img src="https://github.com/bic4907/Overcooked-AI/blob/main/result/online.gif?raw=true" alt="Online MADDPG" width="230"/>|<img src="https://github.com/bic4907/Overcooked-AI/blob/main/result/bc.gif?raw=true" alt="BC" width="230"/>|<img src="https://github.com/bic4907/Overcooked-AI/blob/main/result/offline_rem.gif?raw=true" alt="Offline REM" width="230"/>|


## Acknowledgement

- [Overcooked Environment](https://github.com/HumanCompatibleAI/overcooked_ai)
- [OpenAI's MADDPG Implementation](https://github.com/openai/maddpg)
- [Fujimoto's BCQ Implementation](https://github.com/sfujim/BCQ)
- [Berkelery D4RL evaluations - DDPG_REM](https://github.com/rail-berkeley/d4rl_evaluations/blob/master/crem/rem/DDPG_REM.py)






