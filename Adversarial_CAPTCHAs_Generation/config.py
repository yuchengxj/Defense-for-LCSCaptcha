import torch
from parso import parse
class config:

    # scheme of target captcha
    scheme = 'geetest'

    # batch_size_for_reco_examples

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reco_batch_size = 50
    reco_input_shape = [3, 128, 128]
    reco_height = reco_input_shape[1]
    reco_width = reco_input_shape[2]

    capthca_path = f'./test_set/{scheme}/labelme/test_captcha'

    json_path = f'./test_set/{scheme}/labelme/test_captcha_label'

    class2idx_path = f'./test_set/{scheme}/labelme/class_to_idx.txt'

    reco_output_path = f'./test_set/{scheme}/VOC/JPEGImages_char_noised'



    reco_white_box_model_path = f"./models/reco_model/{scheme}/IncResV2/final_model.pth"

    reco_black_box_models = [(scheme, 'IncV3'), (scheme, 'Res50'), (scheme, 'Vgg16')]

    reco_black_box_model_path = [
        f"./models/reco_model/{scheme}/{model}/final_model.pth" for scheme, model in reco_black_box_models]


    # parameters of M-VNI-CT-FGSM
    # max perturbation of M-VNI-CT-FGSM
    reco_max_eps = 25.5
    # maximum epsilon in M-VNI-CT-FGSM
    central_eps = 76.5

    # number of iteration in M-VNI-CT-FGSM
    reco_iter = 10

    # number neighbours in M-VNI-CT-FGSM
    reco_N = 10

    # momentum of M-VNI-CT-FGSM
    reco_momentum = 1.0

    # number of iteration in M-VNI-CT-FGSM
    reco_beta = 10

    # transformation probablity in M-VNI-CT-FGSM
    reco_prob = 0.7

    # image resize for M-VNI-CT-FGSM
    reco_diver_image_resize = 160


    detect_break_point = 99999
    detect_width = 344
    detect_height = 384
    # 隐含scheme信息
    detect_origin_annotation_path = "./Adversarial_CAPTCHAs_Generation/dataset.txt"
    detect_input_shape = [544, 544]
    detect_batch_size = 2
    detect_num_workers = 2
    detect_classes_path = "./Adversarial_CAPTCHAs_Generation/voc_classes.txt"
    detect_pretrained = False

    # parameters of yolo
    detect_anchors_path_yolo = './Dataset_and_Model_Preparation/Model_Library_Building/yolov5/yolo_anchors.txt'
    detect_anchors_mask_yolo = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    detect_phi = 'l'
    detect_backbone_yolo = 'cspdarknet'
    detect_model_path_yolo = "./models/detect_model/yolo-ep100-loss0.041-val_loss0.039.pth" 
    
    # parameters of frcnn
    detect_model_path_frcnn = "./models/detect_model/frcnn101-ep100-loss0.667-val_loss0.617.pth" 
    detect_anchors_size_frcnn = [4, 16, 32]
    detect_backbone_frcnn = 'resnet101'

    # parameters of ssd
    detect_model_path_ssd = "./models/detect_model/ssd_mb2_ep100_0.9802.pth"  
    detect_anchors_size_ssd = [30, 60, 111, 162, 213, 264, 315]
    detect_backbone_ssd = 'mb2'

    # parameters of SVRE-MI-FGSM
    detect_AA = True
    detect_model_names = ['frcnn', 'yolov5', 'ssd']
    detect_max_eps= 25.5
    detect_num_iter = 10
    detect_momentum = 1.0
    detect_m_svrg = 24
    
    detect_w_yolo = 1
    detect_w_frcnn = 1
    detect_w_ssd = 1

    
