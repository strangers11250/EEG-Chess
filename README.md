## EEG-Chess
An EEG‑based chess game exploring brain–computer interfaces for hands‑free gameplay.

## Catalog
- **Abstract**
- **Authors**
- **Schedule**
- **Development Notes**

### Abstract
EEG‑Chess is a brain–computer interface (BCI) project that lets a player control a chess game using signals recorded from an EEG headset.  
The goal is to explore how well simple EEG features and interaction paradigms (e.g., attention, motor imagery, or event‑related responses) can be used to select pieces, confirm moves, and interact with a standard chess board.  
Beyond building a playable prototype, the project aims to:
- **Investigate** the reliability and latency of EEG‑based control in a turn‑based strategy setting.
- **Visualize** EEG activity during gameplay to give users insight into their brain signals.
- **Document** practical challenges of building an end‑to‑end BCI system (signal quality, artifacts, UX, and calibration).

### Authors (add your name plz)
- **Name**: *Group #24* James Wang, Aidan Ho
- **Contact**: ziw124@ucsd.edu, aiho@ucsd.edu  
- **Course**: COGS 189

### Schedule
A rough timeline to check-in with Professor/TAs.

- **Phase 1 – Background & Design**
  - Review literature on EEG‑based BCIs and relevant signal features.
  - Decide on EEG hardware, acquisition software, and chess interface.
  - Sketch core interaction flow (how a move is selected and confirmed).

- **Phase 2 – Data Pipeline & Prototyping**
  - Implement EEG data acquisition and basic preprocessing (filtering, artifact handling).
  - Build a minimal chess UI that can be controlled programmatically.
  - Connect EEG feature extraction to simple control signals (e.g., binary “select / confirm”).

- **Phase 3 – Integration & UX**
  - Fully integrate EEG control with the chess engine / UI.
  - Design clear feedback for when the system has detected a command.
  - Add basic visualizations of EEG features during gameplay.

- **Phase 4 – Evaluation & Report**
  - Run informal user tests or self‑experiments to assess accuracy and usability.
  - Summarize results, limitations, and future work.
  - Finalize project report, presentation, and repository documentation.

### Development Notes
- **Project goals**
  - Keep the code modular so EEG acquisition, signal processing, and the chess UI can be swapped or improved independently.
  - Make it easy to run small experiments (e.g., comparing different EEG features or control schemes).

- **Tech stack**
  - Programming language: Python 
  - EEG libraries / tools: SSVEP with LDA in EEGLAB.
  - Chess engine / UI: python-chess, pygame

- **Data, ethics, and safety**
  - Only record EEG data from participants who have given informed consent.
  - Avoid storing personally identifying information together with EEG recordings.
  - Clearly document what is being recorded, how it is stored, and how it will be used.
  - This project is for educational / research purposes and is **not** a medical device.

As the implementation solidifies, update this README with concrete commands (how to run the code, file structure, and any specific configuration required for your EEG hardware).
