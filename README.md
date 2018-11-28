# Predicting outcomes of NBA games
## Collaborators
### [Mihika Johorey](https://github.com/mihikajohorey)
### [Pratyush Ranjan Tiwari](https://github.com/PratyushRanjanTiwari)
### [Vineet Reddy](https://github.com/vineetred)

Our goal in this project is to predict the outcome of a National Basket- ball Association game using past game statistics. We use a MultiLayered perceptron to predict Win - Loss as well as game scores, achieving a max- imum accuracy of 89%. We have also iterated over all possibilities of available features to ascertain which statistical variable has the maximum impact on testing accuracy.

Through our project we wanted to determine the possible benefits and limitations of using just game statistics. Feature set importance give us insight on what game strategies are a better predictor of game results in the current NBA seasons.

## File Structure 
    .
    ├── legacy                   # Defunct files. Tinkered around on them.
    ├── finalSubmisison.py       # Win-Loss predictor
    ├── finalSubmisison.ipynb    # Win-Loss predictor Jupyter Notebook
    ├── report.pdf               # Our PDF report
    ├── dataset                  # Testing and training data


## How to run
- Run the command  ``` pip3 install -r install.txt ``` to install all the dependencies (Python 3).

- The script, ```finalSubmission.py``` and ```finalSubmission.ipynb``` are the final scripts. The datasets are already provided. The default model is the Win-Loss predictor. The score line predictor is the ```finalPoints.py```.

- You should be good to go. To exit the script, interrupt using the keyboard.