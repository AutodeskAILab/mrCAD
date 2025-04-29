# mrCAD
## Multimodal Refinement of Computer-Aided Design

Welcome to the mrCAD project page.

The mrCAD project consists of:
- A [paper](arxiv).
- A [dataset](data/mrcad_dataset.csv.zip) of multimodal refinement instructions of 2D Computer-Aided Designs (CADs).
- An [environment]() to benchmark your models on our baseline.
- Preliminary [analyses]() of the dataset.

## The mrCAD Dataset

Download the dataset [here](data/mrcad_dataset.csv.zip).

<div style="text-align: center;">
  <img src="img/task.png" alt="mrcad task" width="50%" />
</div>


The mrCAD dataset was collected by pairing participants in an online communication game.
Over multiple rounds, two players, the *Designer* and the *Maker*, worked together to recreated target designs.
The *Designer* was shown a target design and sent instructions to the *Maker*.
The *Maker* couldn't see the target, so followed the instructions to creating and edit their CAD to match the target.

The mrCAD dataset consists of:
- **15,163 instructions and corresponding executions**,
- across **6,082 unique communication games**,
- performed by **1,092 pairs of participants**,
- and includes **3,166 unique CADs**.

## Dataset partitions

We partition our dataset into three distinct subsets:
- **coverage set**, containing 2249 unique CADs each successfully reconstructed by 1−2 dyads;
- **dense set**, containing 698 unique CADs each successfully reconstructed by at 3−6 dyads;
- **very-dense** set, containing 27 unique CADs successfully reconstructed by at least 30 dyads.


### Instructions

Instructions could include drawing, text, or a combination of both.
The primary data collection goal was to collect examples of *multimodal refinement instructions*, that is, instructions for editing and existing CAD (i.e. round number > 1) that included both text and drawing.

<div style="text-align: center;">
  <img src="img/example_instructions.png" alt="example instructions" width="70%" />
</div>


These include a range of linguistic phenomena such as reference and grounding, as well as extra-linguistic phenomena such as arrows, depiction, and dimensions.


## Environment
