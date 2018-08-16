# Python code

## General
In this folder you can find all the Python code I used to implement my Bayesian model of the co-evolution of language and mindreading (including development, iterated learning and biological evolution). I've organised the python scripts into three different types: those prefixed with `run_`; those prefixed with `unpickle_`, and those without prefix. See below a list of what each of these file types is for: 

* All files *without* a prefix (e.g. `context.py`, `data.py` etc.) are the core components needed to run the different types of simulations. I've separated these out into several modules which implement different parts of the model (e.g. agents, languages, contexts and so on). 

* The files prefixed with `run_` are the four main scripts which import the core modules explained above in order to run a full simulation. The script `run_learner_speaker.py` runs a simulation of a single learner learning from a single speaker (as described in Woensdregt et al., 2016), and `run_learner_pop.py` does the same for a learner receiving data from a population of different speakers. The script `run_pop_iteration.py` runs a simulation of a whole population transmitting languages over generations using iterated learning, in which different selection pressures can be switched on or off. And finally, the script `run_evolvability_analysis.py` loads in the data of a population of literal agents which has already run to convergence, and inserts a pragmatic 'mutant' agent into this population, followed by a simulation of a combined cultural+biological evolution model (where the gene for being 'pragmatic' or 'literal' is transmitted genetically). Each of these main modules starts with a list of parameter settings right after the import statements.
**NOTE** when using these simulation scripts make sure to change the paths (`pickle_file_directory`, `plot_file_directory` and `run_type_dir`) at the top of the `run_` file to match the directories where you want to store your files. 

* The files prefixed with `unpickle_` can load in the various pickle (`.p`) files in which the output of the `run_` scripts are saved, and either plots the data or performs some extra analysis on the data to prepare it for plotting. The module `plots.py` contains a range of different plotting functions, and all the `unpickle_n...` scripts load in a specific set of `.p` files in which the results of the simulations are stored, for plotting or analysis.

## Navigating the code
Throughout these python scripts I used a mixture of object-oriented programming and regular functions, and explained how each function, class or method works using docstrings and comments. I also used long and intelligible variable and function names, which should hopefully make the code relatively easy to read.

## Running large simulations
**NOTE** that some of these simulations take quite a long time to run (learners require a relatively large amount of observations because they are subjected to a joint inference task of simultaneously inferring the perspective and the lexicon of the speaker), so it might be worth outsourcing the running of large simulations to a computer cluster or similar. See more info on that in the folder [code_for_running_on_cluster](https://github.com/marieke-woensdregt/model_coevolution_language_mindreading/tree/master/code_for_running_on_cluster).


### References
Woensdregt, M., Kirby, S., Cummins, C. & Smith, K. (2016). [Modelling the co-development of word learning and perspective-taking.](https://mindmodeling.org/cogsci2016/papers/0222/paper0222.pdf) Proceedings of 38th Annual Meeting of the Cognitive Science Society.

