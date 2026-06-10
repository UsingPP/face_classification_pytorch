class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 7
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    base_path = "/data/wsx1386/repos/faceInterection/arcface-pytorch"

    train_root = base_path + "/data/Datasets/faceImage"
    train_list = base_path + "/data/Datasets/faceImage.txt"
    val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = base_path + '/data/Datasets/faceValidationImage'
    test_list = base_path + '/data/Datasets/faceValidationImage.txt'

    lfw_root = base_path + '/data/Datasets/lfw/lfw-align-128'
    lfw_test_list = base_path + '/data/Datasets/lfw/lfw_pair_list.txt'

    checkpoints_path = base_path + '/models'
    load_model_path = base_path + '/models/resnet18_110.pth'
    test_model_path = base_path + '/models/classifier_metrix_200.pth'
    save_interval = 200

    train_batch_size = 16  # batch size
    test_batch_size = 60
