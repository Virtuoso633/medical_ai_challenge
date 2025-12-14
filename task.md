Test C — Echocardiography Ejection Fraction (EF) Prediction
Problem Statement
Build a minimal pipeline that estimates Left Ventricular Ejection Fraction (EF) from echocardiogram videos using an existing open-source model. You may use either:
EchoNet-Dynamic (Stanford) — https://github.com/echonet/dynamic

Pediatric EchoNet-Dynamic (Pediatric EF prediction) — https://github.com/bryanhe/dynamic/tree/pediatric

Important: EF prediction from echocardiography is a clinically complex task that typically requires highly curated data and specialized model training.
For this challenge, you are not expected to train a new model; you are expected to run inference using an existing pretrained model.
The pediatric model is simpler to set up — you are encouraged to try it first.

Requirements

1. Model Setup
   Download and initialize a pretrained EF model from one of the repositories above.

Ensure the model loads once and can run inference on a given echo video.

Include environment setup instructions (requirements.txt or equivalent).

2. Input Handling
   Accept echocardiogram videos in AVI or MP4 format.

Validate:

File type

File size ≤ 50 MB

Reject invalid files with clear error messages.

3. EF Prediction
   Run the model on the preprocessed video.

Output:

Predicted Ejection Fraction (%)

Optional: predicted end-diastolic and end-systolic frames (if available)

Latency metrics (preprocessing time, inference time)

4. Visualization (Optional but encouraged)
   Generate:

A side-by-side visualization showing sampled frames and predicted key frames
(e.g., ED and ES frames highlighted)

A line plot of EF per frame or the model’s per-frame outputs (if supported)

6. Deliverables
   Your repo submission should include:
   ef_inference.py — command-line script demonstrating EF prediction:

python ef_inference.py --video sample.avi --out results.json

Output:

JSON file containing EF and timing metrics

Any generated visualizations saved to out/ directory

A short explanation in README.md of:

Which model you chose (EchoNet vs Pediatric)

Why EF prediction is clinically and technically challenging

Steps to reproduce inference on another video

Notes for the Candidate
Training a new EF model is NOT required — pretrained inference only.

You may adapt sample inference scripts from the official repos.

EF prediction requires careful handling of echo video quality; document assumptions clearly.

The pediatric model is often easier to run end-to-end; adult EchoNet requires more careful alignment.

If you'd like, I can also generate:
A folder skeleton for your repo

Ready-to-run ef_inference.py boilerplate

README sections for all three tests together
