# Training Atari2600 by Reinforcement Learning
Train Atari2600 and check how it works!

![Breakout](https://user-images.githubusercontent.com/14961526/145698576-00a23e62-5d83-4424-81c5-51aaaf997055.gif)

## How to Setup
You can setup packages on your local env.
```bash
$ make setup
```

or you can run the docker image.
```bash
$ make docker-run
```

## How to Run
```bash
$ python run_atari.py --help

usage: run_atari.py [-h] [--env ENV] [--checkpoint CHECKPOINT] [--n-iters N_ITERS] [--n-workers N_WORKERS] [--gpu]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Atari-2600 env name (Default: Breakout-v0)
  --n-iters N_ITERS     Training iteration number (Default: 10)
  --n-workers N_WORKERS
                        Number of workers for sampling (Default: 4)
  --checkpoint CHECKPOINT
                        Checkpoint path for inference
  --gpu                 Use GPU (Default: False)
  --render              Render env during eval
```

More Atari2600 environments can be found at:
https://gym.openai.com/envs/#atari

#### For Training
```bash
$ python run_atari.py --gpu  # GPU
$ python run_atari.py  # CPU
```

#### For Evaluation
```bash
$ python run_atari.py --render --checkpoint path-to-checkpoint
```

## For Developers
For clean code, you can run formatting or linting.

```bash
$ make format  # formatting
$ make lint  # linting
```
