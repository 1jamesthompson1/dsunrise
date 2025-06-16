# Dynamic Sunrise

This is a new algorithm that extends [sunrise](https://arxiv.org/abs/2007.04938) by making the ensemble dynamic, with removal and addition of new base learners throughout the learning process.

The write up of this algorithm can be found [here](https://github.com/1jamesthompson1/AIML440_report).

After initial attempts at a standalone version ran into too many problems and deadlines loomed the authors original sunrise implementation was used instead. Most of the code is used but has not been deleted due to fear of breaking something.


## Setting up

Once you have the code locally you need to run.

```bash
poetry install
```

Then to run an experiment you can run something like:

```bash
poetry run python OpenAIGym_SAC/examples/dsunrise.py 
```
