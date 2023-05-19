# Defense-for-LCSCaptcha
This is a code release of IJCNN 2023 oral long paper *Fighting Attacks on Large Character Set CAPTCHAs Using Transferable Adversarial Examples*.

### Our code is composed of two parts.
1. `Adversarial CAPTCHAs Generation Part` can generate adversarial CAPTCHAs.

   * `generate_AE_on_character.py` can use `M_VNI_CT_FGSM` to generate adversarial examples in the characters of the CAPTCHA.

   * `generate_AE_on_background.py` can use `SVRE_MI_FGSM` to generate adversarial examples in the background of the CAPTCHA.

   * `config.py` contains the parameter setting and path setting of the adversarial CAPTCHAs generation process.

   * To generate adversarial CAPTCHAs, you can run `generate_AE_on_character.py` first and then run `generate_AE_on_background.py`.

2. `Dataset and Model Preparation Part` contains the synthetic dataset and models.

   * `Model_Library_Building` contains the architecture and training parameters for all character detection and recognition models.

   * `Dataset_Generation` contains scripts to generate the synthetic dataset.

For security and rights considerations, we do not open source the real CAPTCHA dataset and trained attack models. If you need them for research purposes, please contact yfu2668@gmail.com.
