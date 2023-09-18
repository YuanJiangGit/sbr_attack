# SBR_attack

#### Introduction

This code repository is dedicated to assessing the robustness of SBR (Security bug report) detection methods. In this context, we have chosen two state-of-the-art models, FARSEC and LTRWES, as our target models. We aim to evaluate their vulnerability by subjecting them to attacks using four established textual adversarial attack methods: DeepWordBug, TextFooler, PWWS, and TextBugger.

Furthermore, we introduce a novel word-level attack method named SBRAttack, designed to enhance the success rate of attacks on diverse SBR detection models. This repository serves as a platform for conducting experiments, analyzing results, and advancing research in the field of SBR detection and adversarial attacks.

#### Usage

To replicate the experiments and evaluations conducted in this repository, please refer to the provided scripts in the respective folders. The entry points for attacking LTRWES and FARSEC models can be found in the following files:

- `model/AttackLTRWES.py`
- `model/AttackFARSEC.py`

You can run these scripts to initiate the attacks on the LTRWES and FARSEC models as part of your replication process.

#### Dependencies

This project relies on the following dependencies:

- flair==0.6.1.post1
- gensim==3.8.3
- language_tool_python==2.7.1
- lru==0.1
- lru_dict==1.1.8
- matplotlib==2.2.3
- nltk==3.3
- numpy==1.19.4
- pandas==1.1.4
- scikit_learn==1.3.0
- sklearn_evaluation==0.12.0
- tensorflow==2.12.0
- tensorflow_gpu==2.4.1
- tensorflow_hub==0.13.0
- textattack==0.2.15
- torch==1.7.0+cu110
- tqdm==4.62.3
- transformers==4.12.2

Please ensure you have these dependencies installed before running the code. 

#### Contact

For any inquiries, issues, or collaboration opportunities, please feel free to contact us at [jiangyuan@hit.edu.cn](jiangyuan@hit.edu.cn). We welcome your feedback and contributions.