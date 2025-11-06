**Setup**
-

**Install** `pyenv` **:**

On Ubuntu, you can use:
```bash
sudo apt install -y build-essential curl libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

curl https://pyenv.run | bash
```
Add the following at the end of your `~/.bashrc`:

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

**Use `pyenv` to create a local environment with python 3.8.18 .**
```bash
  pyenv install 3.8.18
  pyenv local 3.8.18
  python -m venv .venv
  pip install -r requirements.txt
```

**Run `delay_aware.py` with an MPE `env_id` and model name for the run. The available parameters are:**
```bash
delay_aware.py [-h] [--seed SEED] 
                    [--n_rollout_threads N_ROLLOUT_THREADS]
                    [--n_training_threads N_TRAINING_THREADS]
                    [--buffer_length BUFFER_LENGTH]
                    [--n_episodes N_EPISODES]
                    [--episode_length EPISODE_LENGTH]
                    [--steps_per_update STEPS_PER_UPDATE]
                    [--batch_size BATCH_SIZE]
                    [--n_exploration_eps N_EXPLORATION_EPS]
                    [--init_noise_scale INIT_NOISE_SCALE]
                    [--final_noise_scale FINAL_NOISE_SCALE]
                    [--save_interval SAVE_INTERVAL]
                    [--hidden_dim HIDDEN_DIM]
                    [--lr LR]
                    [--tau TAU]
                    [--agent_alg {MADDPG,DDPG}]
                    [--adversary_alg {MADDPG,DDPG}]
                    [--discrete_action]
                    [--delay_step DELAY_STEP]
                    env_id 
                    model_name
```
**Available `env_id` s available are:**
```python
{
    'simple_speaker_listener',
    'simple_spread',
    'simple_reference',
    'simple_tag',
    'simple_push'
}
```