# Defense-for-LCSCaptcha
This is a code release of paper *Fighting Attacks on Large Character Set CAPTCHAs Using Transferable Adversarial Examples* (IJCNN, 2023).

Our code is composed of two parts.

   ##[Adversarial CAPTCHAs Generation Part](../Adversarial_CAPTCHAs_Generation) can generate adversarial CAPTCHAs.
      ###[generate_AE_on_background.py](../Adversarial_CAPTCHAs_Generation/generate_AE_on_background.py) can use [SVRE_MI_FGSM](../Adversarial_CAPTCHAs_Generation/SVRE_MI_FGSM.py) to generate adversarial examples in the background of the CAPTCHA.
      ###[generate_AE_on_character.py](../Adversarial_CAPTCHAs_Generation/generate_AE_on_character.py) can use [M_VNI_CT_FGSM](../Adversarial_CAPTCHAs_Generation/M_VNI_CT_FGSM.py) to generate adversarial examples in the characters of the CAPTCHA.
      
   ##[Dataset and Model Preparation Part](../Dataset_and_Model_Preparation) contains the synthetic dataset and models.
      ###[Model_Library_Building](../Dataset_and_Model_Preparation/Model_Library_Building) contains the architecture and training parameters for all character detection and recognition models.
      ###[Dataset_Generation](../Dataset_and_Model_Preparation/Dataset_Generation) contains [scripts](../Dataset_and_Model_Preparation/Dataset_Generation/generator.py) to generate the synthetic dataset.
