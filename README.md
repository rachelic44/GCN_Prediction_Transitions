## Transitions Recognition: GCN model

### Backgroud:
Our main goal now is to determine whether a writer is going to change his articles topics. For example, at 2000 he published 200 articles at the field of network, at 2001 he published 70 articles at the same field, but from 2002 and on, he started writing only at the field of operating systems.

In order to achieve this goal, we needed a model to label the data. Means, given 10% labeled data (writers we know their publishing history), and given a graph, the model will label the rest un-labeled publishers with their tags.
For this purpose, we used a GCN model, created using the pytorch geometric.

### data: DBLP data set
The DBLP is an open source dataset available for download at: https://dblp.uni-trier.de/xml/
The DBLP Computer Science bibliography provides comprehensive data about research papers in the Computer Science field.
Our data covers 20 years (1990-2010).
We constructed a network graph so that each researcher in the dataset represents a node in the graph and every two authors are connected with an edge if they publish at least one paper together on the same year. 

The main idea is to create a temporal graph and monitor changes over time. We aimed for a generally stable graph presenting small changes over time.

We aspired to learn from this graph which was generated from a small amount of known information. 
In order to do so, we used 14 different labels (tags) for the data. The tags were determined by the names of the journals that published the articles. 

Two input files were created from the data using xml2txt.py.
As for the data is way to bug to upload, a small dummy is attached, named <edges_little.csv>,  <nodes_little.csv>.










