# Understanding the Attention layer with a toy sequential task

The attention matrix for a sequence of length 6:

![The attention matrix for a sequence of length 6](https://i.imgur.com/2mu13jo.png)


The attention layer has been adopted in numerous sequential and spatial processing. David Blei characterized the attention layer as highlighting features that are relevant for predicting and outcome can depend on the entire feature set as well. 

François Fleuret came up with a [toy sequential task](https://twitter.com/francoisfleuret/status/1262639062785105922) that's great for learning about attention mechanisms. I find notebooks to be the ideal playground for fiddling with ideas and in this repo I take François' work to hack around and explore. 

The `demo.ipynb` notebook goes through his code and tries to explain how the various components work. I tried to strip away his original code to what is most useful for learning and exploring. 
