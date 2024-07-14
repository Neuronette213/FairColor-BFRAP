# FairColor-BFRAP
This repository implements FairColor from our paper, FairColor: An Efficient Algorithm for the Balanced and Fair Reviewer Assignment Problem 

## Datasets
Four instances from real conferences: MIDL'18 (Instance 30), CVPR'17 (Instance 40), CVPR'18 (Instance 50), ICA2IT'19 (Instance 60)
Synthetic datasets: CVPR'18Extd_2 (Instance 70), ICLR'18 (Instance 80), CVPR'17Extd_4 (Instance 90), CVPR'17Extd_4 (Instance 100), MIDL'18Extd_4 (Instance 110), ICA2IT'19Extd_4 (Instance 120), and ICLR'18Extd_4 (Instance 130)
Special Instances C1 (Instance 1), C2 (Instance 2), C3 (Instance 3), C4 (Instance 4), and C5 (Instance 5)
are available here 
[Dataset Folder](https://drive.google.com/drive/folders/1dv10bSwwyUIAHLS5o9WAcboml6yd7GtL)

MIDL'18, CVPR'17, and CVPR'18 datasets have been made available by Kobren et al.: [Paper Matching with Local Fairness Constraints](https://github.com/iesl/fair-matching)

ICLR'18 dataset has been made available by Xu et al.:[On Strategyproof Conference Peer Review](https://github.com/xycforgithub/StrategyProof_Conference_Review)

C1 to C5 instances have been generated according to Stelmakh et al.'s paper: [PeerReview4All:
Fair and Accurate Reviewer Assignment in Peer Review](https://www.jmlr.org/papers/volume22/20-190/20-190.pdf)

The conference organizers have provided the ICA2IT'19 dataset. Similarity scores have been calculated using the approach proposed by Medakene et al.: [A New Approach for Computing the Matching Degree in the Paper-to-Reviewer Assignment Problem](https://doi.org/10.1109/ICTAACS48474.2019.8988127)

[Dataset_statistics.py](https://github.com/Neuronette213/FairColor-BFRAP/blob/main/Dataset_statistics.py) displays dataset statistics.

## Experiments
We compare FairColor to FairFlow and FairIR. Kobren et al. designed both algorithms to find fair and efficient assignments. They are adapted here to load balance constraints. 

[Results_comparison.py](https://github.com/Neuronette213/FairColor-BFRAP/blob/main/Results_comparison.py) provides the FairColor, FairFlow, and FairIR comparison results. 
