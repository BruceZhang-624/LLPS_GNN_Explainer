# LLPS_GNN_Explainer

This repository is mainly for the final project of **Trustworthy Deep Learning (CPSC 680)**

## Abstract
Liquid–liquid phase separation (LLPS) as the underlying principle represents some of the underlying phenomenon of biological cellular activities, including membraneless organelles. Formation of membrane-bound organelles allow the enclosure of specific proteins controlled by genes without mutual adverse effects. Recent studies hints that membraneless organelles contain an representation of proteins with intrinsically structural and characterized residues which are important drivers of phase-separation behavior. To decode the residue structural determinants of protein phase separation is important for understanding the biochemistry of bio-molecular condensates. Existing studies are devoted to constructing theoretical models based on experimental results, however, their analysis of the independent variables is often discrete and difficult to provide a more in-depth explanation of the more specific physicochemical properties of the residue. In this proposal, we propose to give a new scheme for protein LLPS classification based on state-of-the-art derivative technologies of graph neural network and introduce explainable deep learning for investigating residue structural determinants of LLPS.

![image](/images/workflow.JPEG)

### Environment and Dependencies

 * environment.yml

### Usage
1. TO generate the Graph, please refer to `.py`.
2. TO training the model, please use `.ip`


### Results
One example of comparsion between k-means and GNN Explainer:
![image](/images/example.png)
