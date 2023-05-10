# LLPS_GNN_Explainer

Ling Han, Jiazhen Zhang, Yifei He

This repository is mainly for the final project of **Trustworthy Deep Learning (CPSC 680)**

## Abstract
Liquid–liquid phase separation (LLPS) as the underlying principle represents some of the underlying phenomenon of biological cellular activities, including membraneless organelles. Formation of membrane-bound organelles allow the enclosure of specific proteins controlled by genes without mutual adverse effects. Recent studies hints that membraneless organelles contain an representation of proteins with intrinsically structural and characterized residues which are important drivers of phase-separation behavior. To decode the residue structural determinants of protein phase separation is important for understanding the biochemistry of bio-molecular condensates. Existing studies are devoted to constructing theoretical models based on experimental results, however, their analysis of the independent variables is often discrete and difficult to provide a more in-depth explanation of the more specific physicochemical properties of the residue. In this proposal, we propose to give a new scheme for protein LLPS classification based on state-of-the-art derivative technologies of graph neural network and introduce explainable deep learning for investigating residue structural determinants of LLPS.

![image](/images/workflow.JPEG)

### Environment and Dependencies

 * environment.yml

### Usage
1. TO generate the Graph, please refer to `/code/GNN_createGraph.py`. PS: In this task, the generation of the graph is not our foucs. However, we provide a way of creating the graph data.
2. TO train the model, please use each of notebook `/code/*.ipynb` (except for `kmeans`) for training. We offer four types of Graph Neural Networks, including `GNN`, `GCN`, `GAT`, and `GIN`. The detailed implementation is in `/code/GNN_core.py`.
3. TO realize the Explanation, please use each of notebook `/code/*.ipynb`. We implement three types of explanation methonds, including `GNNExplainer`, `PGExplainer`, and `Integrated Gradients`. The baseline method we use here is `k-means`.


## Results
[to do]

### Results
One example of comparsion between k-means and GNN Explainer:

![image](/images/example.png)


## Contribution 
**Ling Han**: Methodology. **Jiazhen Zhang**: Coding. **Yifei He**: Analysis.
