# Diffusion-based Generative Fiber Orientation Restoration from Severe Distortions in Diffusion-weighted MRI (FOD_Diffusion)
Fiber Orientation Distributions (FODs) is a widely used model to represent diffusion MRI (dMRI) data. However, susceptibility-induced distortion can cause signal loss and  corrupted reconstruction of FODs in multiple brain regions even after the state-of-the-art distortion correction, which can severely affect fiber tracking and connectivity analysis. Generative models, such as diffusion models, have been successfully applied in various image restoration tasks. Their application to FODs, however, poses unique challenges since FODs are four-dimensional images represented by spherical harmonics (SPHARM), with the fourth dimension exhibiting order-related dependency. In this paper, we propose a novel diffusion model for FOD restoration. We use volume-order encoding to enhance the ability of the diffusion model to generate individual FOD volumes at all SPHARM orders. Moreover, we add cross-attention features extracted across all SPHARM orders to capture the order-related dependency of FOD volumes. We also condition the diffusion model with low-distortion FODs surrounding high-distortion areas to maintain geometric coherence.

Figure 1 shows the workflow of our method.

<p align="center">
  <img width="3703" height="1903" alt="Figure_proposed_method" src="https://github.com/user-attachments/assets/751f8a2c-1a36-4556-a686-d7933cbfce54" />
  <br/>
  <em>Figure 1. Workflow of our proposed method.</em>
</p>

If you think our work useful, please cite it as:

``bibtex
@article{huang2024diffusion,
  title={Diffusion Model-based FOD Restoration from High Distortion in dMRI},
  author={Huang, Shuo and Zhong, Lujia and Shi, Yonggang},
  journal={ArXiv},
  pages={arXiv--2406},
  year={2024}
}
