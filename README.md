Developed by Michael Gold (michael.gold@ll.mit.edu)

Welcome to blob finding with ML! 
This project was developed for the purpose of identifying magnetic flux trapped in superconducting materials. Conventional methods, like blob_log, are somewhat effective but require frequent retuning. This project is intended to provide a more universal pipeline for vortex localization and a more convenient method of correcting errors. 

Below is an outline of the project. See comments at the top of each file for more specific instructions.

Configuring Project Environment:
    1. I used uv to manage this project, though others may work fine too
    2. (optional) Add the required index and source sections in pyproject.toml if gpu version of pytorch is desired
    3. Add dependencies to the environment
    4. (files needed) - uv.lock, pyproject.toml, .python-version, .venv

Training a CNN:
    A. Label training data
        1. Crop Data.py - create ROIs from full experiment FOVs
        2. Label Data.py - GUI to identify the center of each vortex
        3. Make Heatmaps.py - create heatmap label for each ROI from its coresponding vortex coordinate list
        4. (files needed) - Crop Data.py, Label Data.py, Make Heatmaps.py
    B. Train a model
        1. config.py - set parameters for training
        2. main.py - initiates training the model using the labelled data
        3. (optional) model_results.py - view the resutls of training if desired
        4. (files needed) - main.py, config.py, functions.py, model.py, model_results.py

Vortex Identification:
    1. (optional) use_model.py - spot check one image at a time to see the new model in action
    2. vortex_finder.py - run a folder of data through the model as a first pass
    3. vortex_editor.py - make any necessary adjustments to the identified vortices and fit the results
    4. (files needed) - vortex_finder.py, vortex_editor.py, use_model.py, model.py, {modelname}.pt


Note: This project was created with the assistance of ChatGPT
